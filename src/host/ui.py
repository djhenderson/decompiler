# -*- coding: utf-8 -*-

from __future__ import print_function
import traceback

try:
  import idaapi # try importing ida's main module.

  print('INFO: Using IDA backend.')
  from .ida.ui import *
except BaseException as e:
  print('WARNING: IDA backend is not available.')
  print(repr(e))
  traceback.print_exc()
