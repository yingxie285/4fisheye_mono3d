from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
from collections import OrderedDict

import cv2
import numpy as np
import pycocotools.coco as coco
import torch.utils.data as data


class KITTI(data.Dataset):
  num_classes = 13
  default_resolution = [960, 1408]
  mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
  std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
  class_name = [
    'car', 'truck', 'bus', 'construction_vehicle', 'pedestrian',
    'motor', 'bicycle', 'animal', 'traffic_cone', 'barrier',
    'stopper', 'trash_bin', 'sign'
  ]

  def __init__(self, opt, split):
    super(KITTI, self).__init__()
    self.split = split
    self.opt = opt
    self.alpha_in_degree = False

    self.data_dir = os.path.join(opt.data_dir, 'demo_data', 'trainval')
    self.img_dir = os.path.join(self.data_dir, 'images')
    self.train_annot_path = os.path.join(
      self.data_dir, 'annotations', 'ip42_train_all.json')
    self.val_annot_path = os.path.join(
      self.data_dir, 'annotations', 'ip42_val_all.json')
    self.annot_path = self.train_annot_path if split == 'train' else \
      self.val_annot_path

    self.max_objs = 100
    self.cat_ids = {i: i - 1 for i in range(1, 14)}

    self.frame_group_size = max(1, getattr(opt, 'frame_group_size', 1))
    self.camera_order = [
      camera.strip() for camera in getattr(
        opt, 'camera_order', 'back,front,left,right').split(',')
      if camera.strip()
    ]
    self.required_cameras = self.camera_order[:self.frame_group_size]
    self.camera_rank = {
      camera_name: idx for idx, camera_name in enumerate(self.camera_order)
    }
    self.multiview_enabled = self.frame_group_size > 1
    self.multiview_nms_iou = getattr(opt, 'multiview_nms_iou', 0.1)

    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array(
      [0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
    self._eig_vec = np.array([
      [-0.58752847, -0.69563484, 0.41340352],
      [-0.5832747, 0.00994535, -0.81221408],
      [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)

    print('==> initializing ip42 {} data.'.format(split))
    dataset_dict = self._load_dataset_dict(split)
    self.coco = coco.COCO()
    self.coco.dataset = dataset_dict
    self.coco.createIndex()

    self.image_infos_by_id = {}
    self.image_infos_by_stem = {}
    self.image_infos_by_file_name = {}
    self.images, self.frame_ids, dropped_frames = self._build_image_index()
    self.num_samples = len(self.images)

    if dropped_frames:
      print(
        'Dropped {} incomplete frame groups for split {} (need cameras: {}).'
        .format(len(dropped_frames), split, ','.join(self.required_cameras))
      )

    if self.multiview_enabled and split == 'train' and \
        getattr(opt, 'batch_size', 1) % self.frame_group_size != 0:
      print(
        'Warning: batch_size {} is not divisible by frame_group_size {}.'
        ' Multi-camera batches will cross frame boundaries.'.format(
          opt.batch_size, self.frame_group_size)
      )

    print(
      'Loaded {} {} samples across {} frames'.format(
        split, self.num_samples, len(self.frame_ids))
    )

  def __len__(self):
    return self.num_samples

  def _to_float(self, x):
    return float('{:.2f}'.format(x))

  def _file_stem(self, file_name):
    return os.path.splitext(os.path.basename(file_name))[0]

  def _extract_frame_id(self, img_info):
    frame_id = img_info.get('frame_id')
    if frame_id is not None and frame_id != '':
      return str(frame_id)
    stem = self._file_stem(img_info['file_name'])
    parts = stem.rsplit('__', 2)
    return parts[-1] if len(parts) == 3 else stem

  def _extract_camera_name(self, img_info):
    camera_name = img_info.get('camera_name')
    if camera_name:
      return str(camera_name)
    stem = self._file_stem(img_info['file_name'])
    parts = stem.rsplit('__', 2)
    return parts[-2] if len(parts) == 3 else 'mono'

  def _frame_sort_key(self, frame_id):
    return int(frame_id) if str(frame_id).isdigit() else str(frame_id)

  def _image_sort_key(self, img_info):
    frame_id = self._extract_frame_id(img_info)
    camera_name = self._extract_camera_name(img_info)
    return (
      self._frame_sort_key(frame_id),
      self.camera_rank.get(camera_name, len(self.camera_rank)),
      self._file_stem(img_info['file_name'])
    )

  def _load_json(self, path):
    with open(path, 'r', encoding='utf-8') as fp:
      return json.load(fp)

  def _load_dataset_dict(self, split):
    return self._prepare_dataset_dict(self._load_json(self.annot_path), split)

  def _prepare_dataset_dict(self, dataset, split):
    new_images = []
    old_to_new_image_ids = {}
    seen_file_names = set()

    for img_info in dataset.get('images', []):
      file_name = img_info['file_name']
      if file_name in seen_file_names:
        continue
      seen_file_names.add(file_name)

      new_img_info = dict(img_info)
      new_img_info['frame_id'] = self._extract_frame_id(img_info)
      new_img_info['camera_name'] = self._extract_camera_name(img_info)
      new_img_info['source_split'] = split
      new_img_info['original_id'] = img_info['id']
      new_img_info['id'] = len(new_images) + 1
      new_images.append(new_img_info)
      old_to_new_image_ids[img_info['id']] = new_img_info['id']

    new_annotations = []
    for ann in dataset.get('annotations', []):
      new_image_id = old_to_new_image_ids.get(ann['image_id'])
      if new_image_id is None:
        continue
      new_ann = dict(ann)
      new_ann['id'] = len(new_annotations) + 1
      new_ann['image_id'] = new_image_id
      new_annotations.append(new_ann)

    return {
      'images': new_images,
      'annotations': new_annotations,
      'categories': dataset.get('categories', []),
      'info': dataset.get('info', {}),
      'licenses': dataset.get('licenses', [])
    }

  def _build_image_index(self):
    image_ids = self.coco.getImgIds()
    image_infos = self.coco.loadImgs(ids=image_ids)
    image_infos = sorted(image_infos, key=self._image_sort_key)

    for img_info in image_infos:
      self.image_infos_by_id[img_info['id']] = img_info
      self.image_infos_by_stem[self._file_stem(img_info['file_name'])] = img_info
      self.image_infos_by_file_name[img_info['file_name']] = img_info

    frame_to_images = OrderedDict()
    for img_info in image_infos:
      frame_id = self._extract_frame_id(img_info)
      frame_to_images.setdefault(frame_id, []).append(img_info)

    ordered_image_ids = []
    ordered_frame_ids = []
    dropped_frames = []
    self.frame_to_image_ids = OrderedDict()

    for frame_id, frame_images in frame_to_images.items():
      if self.multiview_enabled:
        grouped_image_ids = self._build_complete_frame_group(frame_images)
        if grouped_image_ids is None:
          dropped_frames.append(frame_id)
          continue
      else:
        grouped_image_ids = [
          img_info['id']
          for img_info in sorted(frame_images, key=self._image_sort_key)
        ]

      ordered_frame_ids.append(frame_id)
      ordered_image_ids.extend(grouped_image_ids)
      self.frame_to_image_ids[frame_id] = grouped_image_ids

    if self.multiview_enabled and not ordered_frame_ids:
      raise RuntimeError(
        'No complete {}-camera frame groups were found for split {}.'
        .format(self.frame_group_size, self.split)
      )

    return ordered_image_ids, ordered_frame_ids, dropped_frames

  def _build_complete_frame_group(self, frame_images):
    camera_to_image = {}
    for img_info in frame_images:
      camera_name = self._extract_camera_name(img_info)
      if camera_name not in camera_to_image:
        camera_to_image[camera_name] = img_info

    if any(camera_name not in camera_to_image for camera_name in self.required_cameras):
      return None

    return [
      camera_to_image[camera_name]['id']
      for camera_name in self.required_cameras
    ]

  def _resolve_result_image_info(self, result_key):
    if hasattr(result_key, 'item'):
      result_key = result_key.item()

    if isinstance(result_key, (int, np.integer)):
      return self.image_infos_by_id.get(int(result_key))

    key = str(result_key)
    if key in self.image_infos_by_stem:
      return self.image_infos_by_stem[key]
    if key in self.image_infos_by_file_name:
      return self.image_infos_by_file_name[key]
    if key.isdigit():
      return self.image_infos_by_id.get(int(key))
    return None

  def _as_det_array(self, dets):
    if dets is None:
      return np.zeros((0, 13), dtype=np.float32)
    det_array = np.asarray(dets, dtype=np.float32)
    if det_array.size == 0:
      return np.zeros((0, 13), dtype=np.float32)
    if det_array.ndim == 1:
      det_array = det_array.reshape(1, -1)
    return det_array

  def _bev_rotated_rect(self, det):
    center_x = float(det[8])
    center_y = float(det[9])
    width = max(abs(float(det[6])), 1e-3)
    length = max(abs(float(det[7])), 1e-3)
    yaw = math.degrees(float(det[11]))
    return ((center_x, center_y), (length, width), yaw)

  def _bev_iou(self, det_a, det_b):
    rect_a = self._bev_rotated_rect(det_a)
    rect_b = self._bev_rotated_rect(det_b)
    inter_type, inter_points = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if inter_type == cv2.INTERSECT_NONE or inter_points is None:
      return 0.0

    inter_area = abs(cv2.contourArea(inter_points))
    area_a = rect_a[1][0] * rect_a[1][1]
    area_b = rect_b[1][0] * rect_b[1][1]
    union = max(area_a + area_b - inter_area, 1e-6)
    return float(inter_area / union)

  def _multiview_nms(self, dets):
    dets = self._as_det_array(dets)
    if dets.shape[0] <= 1 or self.multiview_nms_iou <= 0:
      return dets

    order = np.argsort(-dets[:, -1])
    keep = []

    while order.size > 0:
      current = order[0]
      keep.append(current)
      remaining = []
      for candidate in order[1:]:
        if self._bev_iou(dets[current], dets[candidate]) < self.multiview_nms_iou:
          remaining.append(candidate)
      order = np.asarray(remaining, dtype=np.int64)

    return dets[keep]

  def _merge_frame_results(self, results):
    merged_results = OrderedDict(
      (frame_id, {cls_ind: [] for cls_ind in range(1, self.num_classes + 1)})
      for frame_id in self.frame_ids
    )
    unresolved_keys = []

    for result_key, image_results in results.items():
      img_info = self._resolve_result_image_info(result_key)
      if img_info is None:
        unresolved_keys.append(str(result_key))
        continue

      frame_id = self._extract_frame_id(img_info)
      if frame_id not in merged_results:
        merged_results[frame_id] = {
          cls_ind: [] for cls_ind in range(1, self.num_classes + 1)
        }

      for cls_ind in range(1, self.num_classes + 1):
        det_array = self._as_det_array(image_results.get(cls_ind, []))
        if det_array.size > 0:
          merged_results[frame_id][cls_ind].append(det_array)

    if unresolved_keys:
      print(
        'Warning: {} result keys could not be matched back to images.'.format(
          len(unresolved_keys))
      )

    final_results = OrderedDict()
    for frame_id, class_results in merged_results.items():
      final_results[frame_id] = {}
      for cls_ind in range(1, self.num_classes + 1):
        if class_results[cls_ind]:
          dets = np.concatenate(class_results[cls_ind], axis=0)
          dets = self._multiview_nms(dets) if self.multiview_enabled else dets
          final_results[frame_id][cls_ind] = dets
        else:
          final_results[frame_id][cls_ind] = np.zeros((0, 13), dtype=np.float32)
    return final_results

  def convert_eval_format(self, all_bboxes):
    pass

  def save_results(self, results, save_dir):
    lidar_results_dir = os.path.join(save_dir, 'results_lidar')
    os.makedirs(lidar_results_dir, exist_ok=True)

    frame_results = self._merge_frame_results(results)
    ordered_frame_ids = list(self.frame_ids)
    for frame_id in frame_results.keys():
      if frame_id not in self.frame_to_image_ids:
        ordered_frame_ids.append(frame_id)

    for frame_id in ordered_frame_ids:
      lidar_out_path = os.path.join(lidar_results_dir, '{}.txt'.format(frame_id))
      with open(lidar_out_path, 'w') as f_lidar:
        for cls_ind in range(1, self.num_classes + 1):
          det_array = self._as_det_array(frame_results.get(frame_id, {}).get(cls_ind, []))
          class_name = self.class_name[cls_ind - 1]
          for det in det_array:
            lidar_det = det[:13]
            f_lidar.write('{} 0.0 0'.format(class_name))
            for value in lidar_det:
              f_lidar.write(' {:.2f}'.format(value))
            f_lidar.write('\n')

  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    eval_tool = os.path.abspath(os.path.join(
      os.path.dirname(__file__), '..', '..', '..',
      'tools', 'kitti_eval', 'evaluate_object_3d_offline'))
    label_dir = os.path.abspath(os.path.join(
      os.path.dirname(__file__), '..', '..', '..',
      '..', 'data', 'kitti', 'training', 'label_val'))
    results_dir = os.path.join(save_dir, 'results')

    if os.path.isfile(eval_tool) and os.path.isdir(label_dir) and \
        os.path.isdir(results_dir):
      os.system('{} {} {}/'.format(eval_tool, label_dir, results_dir))
    else:
      print(
        'Saved frame-level lidar results to {}. Skipped legacy KITTI camera '
        'evaluator because the required camera-space paths were not found.'
        .format(os.path.join(save_dir, 'results_lidar'))
      )
