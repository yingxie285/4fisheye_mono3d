from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

# from object_det.external.nms import soft_nms
from object_det.opts import opts
from object_det.logger import Logger
from object_det.utils.utils import AverageMeter
from object_det.datasets.dataset_factory import dataset_factory
from object_det.detectors.detector_factory import detector_factory

def load_ddd_calib(calib_dir, file_name, fallback_calib=None):
  calib_path = os.path.join(calib_dir, os.path.splitext(file_name)[0] + '.txt')
  p_cyl = None
  k_cyl = None
  r_final = None
  tr_velo_to_cam = None

  if fallback_calib is not None:
    p_cyl = np.array(fallback_calib, dtype=np.float32).reshape(3, 4)

  if os.path.exists(calib_path):
    with open(calib_path, 'r') as f:
      for raw_line in f:
        line = raw_line.strip()
        if not line or ':' not in line:
          continue
        key, value = line.split(':', 1)
        values = np.array(value.strip().split(), dtype=np.float32)
        if key == 'P0' and p_cyl is None and values.size == 9:
          p_cyl = np.concatenate(
            [values.reshape(3, 3), np.zeros((3, 1), dtype=np.float32)], axis=1)
        elif key == 'K_cyl' and values.size == 9:
          k_cyl = values.reshape(3, 3)
        elif key == 'R_final' and values.size == 9:
          r_final = values.reshape(3, 3)
        elif key == 'Tr_velo_to_cam' and values.size == 12:
          tr_velo_to_cam = values.reshape(3, 4)

  if p_cyl is None and k_cyl is not None:
    p_cyl = np.concatenate(
      [k_cyl, np.zeros((3, 1), dtype=np.float32)], axis=1)

  if p_cyl is None:
    raise ValueError('Missing cylindrical calibration for {}'.format(file_name))

  if k_cyl is None:
    k_cyl = p_cyl[:, :3]

  if r_final is None or tr_velo_to_cam is None:
    return p_cyl

  stacked_calib = np.zeros((9, 4), dtype=np.float32)
  stacked_calib[:3] = np.concatenate(
    [k_cyl, np.zeros((3, 1), dtype=np.float32)], axis=1)
  stacked_calib[3:6, :3] = r_final
  stacked_calib[6:9] = tr_velo_to_cam
  return stacked_calib



class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.calib_dir = os.path.join(dataset.data_dir, 'calibs')
    self.pre_process_func = pre_process_func
    self.opt = opt

  def __getitem__(self, index):
    # img_id = self.images[index]
    # img_info = self.load_image_func(ids=[img_id])[0]
    # img_path = os.path.join(self.img_dir, img_info['file_name'])

    #-----------------------------------------------------------
    coco_id = self.images[index]   # 原始COCO id（可能是大整数）
    img_info = self.load_image_func(ids=[coco_id])[0]
    # 0317 用文件名作为最终ID（最安全）
    file_name = img_info['file_name']   # 1706192392264530944.jpg
    img_id = file_name.split('.')[0]    # "1706192392264530944"
    img_path = os.path.join(self.img_dir, file_name)
    #-----------------------------------------------------------

    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in self.opt.test_scales:
      if self.opt.task == 'ddd':
        calib = load_ddd_calib(
          self.calib_dir, file_name, img_info.get('calib'))
        images[scale], meta[scale] = self.pre_process_func(
          image, scale, calib)
      else:
        images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.images)


def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]

  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)

  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process),
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    ret = detector.run(pre_processed_images)
    # results[img_id.numpy().astype(np.int32)[0]] = ret['results']
    # 0317 ==========
    img_id = img_id[0]  # batch_size=1

    # 如果是 tensor → 转 Python 类型
    if hasattr(img_id, 'item'):
        img_id = img_id.item()

    # 强制转字符串（避免任何溢出问题）
    img_id = str(img_id)

    results[img_id] = ret['results']
    # ===================
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)


def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]

  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])
    result_key = os.path.splitext(img_info['file_name'])[0]

    if opt.task == 'ddd':
      calib = load_ddd_calib(
        os.path.join(dataset.data_dir, 'calibs'),
        img_info['file_name'], img_info.get('calib'))
      ret = detector.run(img_path, calib)
    else:
      ret = detector.run(img_path)

    results[result_key] = ret['results']

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)


def run(args=''):
  opt = opts().parse(args)
  if opt.not_prefetch_test:
    test(opt)
  else:
    prefetch_test(opt)
