from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.utils.data

from object_det.opts import opts
from object_det.models.model import create_model, load_model, save_model
from object_det.logger import Logger
from object_det.datasets.dataset_factory import get_dataset
from object_det.trains.train_factory import train_factory
import time

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = False
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
  print('Using GPUs:', opt.gpus)
  
  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'),
      batch_size=1,
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'),
      batch_size=opt.batch_size,
      shuffle=False,  #True
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  ) #划分数据集的时候已经按场景打乱了顺序

  print('Starting training...')
  best = 1e10
  train_start_time = time.time()   # 新增：记录训练开始时间
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):  #num_epochs=50
    epoch_start = time.time()   # 新增：记录epoch开始
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    
    # ===== 新增：计算剩余时间 =====
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - train_start_time
    avg_epoch_time = elapsed / (epoch - start_epoch)
    remaining_epoch = opt.num_epochs - epoch
    eta_seconds = avg_epoch_time * remaining_epoch
    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
    # ============================
    
    # logger.write('epoch: {} |'.format(epoch))
    logger.write('epoch: {} | ETA {} | '.format(epoch, eta_str))  # 0307修改
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)  #传的是epoch，loss曲线的横坐标是epoch
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()


def run(args=''):
  opt = opts().parse(args)
  main(opt)

