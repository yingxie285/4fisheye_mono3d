from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from object_det.opts import opts
from object_det.cli.test import prefetch_test, test

if __name__ == '__main__':
  opt = opts().parse()
  if opt.not_prefetch_test:
    test(opt)
  else:
    prefetch_test(opt)
