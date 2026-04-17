from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds
from .ddd_utils import ddd2locrot


def wrap_to_pi(angle):
  return (angle + np.pi) % (2 * np.pi) - np.pi


def unpack_cyl_calib(calib):
  calib = np.asarray(calib, dtype=np.float32)
  calib_info = {
    'K_cyl': calib[:3, :3],
    'R_final': None,
    'Tr_velo_to_cam': None,
    'cam2lidar': None
  }
  if calib.ndim == 2 and calib.shape[0] >= 9 and calib.shape[1] >= 4:
    calib_info['R_final'] = calib[3:6, :3]
    calib_info['Tr_velo_to_cam'] = calib[6:9, :4]
    tr_h = np.eye(4, dtype=np.float32)
    tr_h[:3, :4] = calib_info['Tr_velo_to_cam']
    calib_info['cam2lidar'] = np.linalg.inv(tr_h)
  return calib_info


def cyl_location_to_lidar(location_cyl, rotation_y_cyl, calib_info):
  if calib_info['R_final'] is None or calib_info['cam2lidar'] is None:
    return None, None

  location_cam = calib_info['R_final'].dot(location_cyl.astype(np.float32))
  location_cam_h = np.concatenate(
    [location_cam, np.ones(1, dtype=np.float32)], axis=0)
  location_lidar = calib_info['cam2lidar'].dot(location_cam_h)[:3]

  # Your cylindrical labels use local +Z as the object's forward axis.
  forward_cyl = np.array([
    np.sin(rotation_y_cyl),
    0.0,
    np.cos(rotation_y_cyl)
  ], dtype=np.float32)
  forward_cam = calib_info['R_final'].dot(forward_cyl)
  forward_lidar = calib_info['cam2lidar'][:3, :3].dot(forward_cam)
  rotation_lidar = wrap_to_pi(
    np.arctan2(forward_lidar[1], forward_lidar[0]))

  return location_lidar.astype(np.float32), np.float32(rotation_lidar)


def get_pred_depth(depth):
  return depth


def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + (0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)


def ddd_post_process_2d(dets, c, s, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  include_wh = dets.shape[2] > 16
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
      dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
    classes = dets[i, :, -1]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :3].astype(np.float32),
        get_alpha(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
        get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
        dets[i, inds, 12:15].astype(np.float32)], axis=1)
      if include_wh:
        top_preds[j + 1] = np.concatenate([
          top_preds[j + 1],
          transform_preds(
            dets[i, inds, 15:17], c[i], s[i], (opt.output_w, opt.output_h))
          .astype(np.float32)], axis=1)
    ret.append(top_preds)
  return ret


def ddd_post_process_3d(dets, calibs):  #0413 只保存最终的results_lidar
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  for i in range(len(dets)):
    preds = {}
    calib_info = unpack_cyl_calib(calibs[i])
    k_cyl = calib_info['K_cyl']
    f = k_cyl[0, 0]
    u0 = k_cyl[0, 2]
    v0 = k_cyl[1, 2]

    for cls_ind in dets[i].keys():
      preds[cls_ind] = []
      for j in range(len(dets[i][cls_ind])):
        center = dets[i][cls_ind][j][:2]
        score = dets[i][cls_ind][j][2]
        alpha = dets[i][cls_ind][j][3]
        rho = dets[i][cls_ind][j][4]
        dimensions = dets[i][cls_ind][j][5:8]
        wh = dets[i][cls_ind][j][8:10]

        u, v = center
        phi = (u - u0) / f
        psi = (v - v0) / f

        dir_vec = np.array([
          np.sin(phi),
          psi,
          np.cos(phi)
        ], dtype=np.float32)
        dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-6)

        location_cyl = dir_vec * rho
        rotation_y_cyl = wrap_to_pi(alpha + phi)

        bbox = [
          u - wh[0] / 2,
          v - wh[1] / 2,
          u + wh[0] / 2,
          v + wh[1] / 2
        ]

        location_lidar, rotation_lidar = cyl_location_to_lidar(
          location_cyl, rotation_y_cyl, calib_info)

        if location_lidar is None:
          location_out = location_cyl
          rotation_out = rotation_y_cyl
        else:
          location_out = location_lidar
          rotation_out = rotation_lidar

        # Keep a single 13-dim KITTI-style output whose 3D fields are already
        # in lidar coordinates, with score as the last column.
        pred = [alpha] + bbox + dimensions.tolist() + \
               location_out.tolist() + [float(rotation_out), score]

        preds[cls_ind].append(pred)

      preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)

    ret.append(preds)

  return ret


def ddd_post_process(dets, c, s, calibs, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  dets = ddd_post_process_2d(dets, c, s, opt)
  dets = ddd_post_process_3d(dets, calibs)
  return dets


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
      dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
      dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret


def multi_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5],
       pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret