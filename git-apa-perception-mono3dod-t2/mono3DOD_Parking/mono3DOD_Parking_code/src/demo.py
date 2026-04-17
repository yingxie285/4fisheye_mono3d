from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import ssl
import urllib.request

# 全局禁用 SSL 证书验证
ssl._create_default_https_context = ssl._create_unverified_context

# 现在你的原有代码可以直接运行，无需额外传 context
try:
    # 替换成你要访问的实际 URL
    response = urllib.request.urlopen('https://www.baidu.com')
    print("请求成功！状态码：", response.getcode())
except Exception as e:
    print("请求失败：", str(e))

from object_det.opts import opts
from object_det.cli.demo import demo
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
