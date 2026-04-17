from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from object_det.opts import opts
from object_det.cli.train import main as train_main


def main(opt):
  return train_main(opt)

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
