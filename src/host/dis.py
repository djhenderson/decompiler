# -*- coding: utf-8 -*-

from __future__ import print_function
import traceback

available_disassemblers = {}

try:
  import idaapi # try importing ida's main module.
  import ida.dis

  print('INFO: Using IDA backend.')
  available_disassemblers['ida'] = ida.dis
except ImportError as e:
  print('WARNING: IDA backend is not available.')
  pass
except BaseException as e:
  print('WARNING: IDA backend is not available.')
  print(repr(e))
  traceback.print_exc()

try:
  import capstone # try importing capstone.
  from .capstone import dis

  print('INFO: Using Capstone backend.')
  available_disassemblers['capstone'] = dis
except ImportError as e:
  print('WARNING: Capstone backend is not available.')
  print(repr(e))
  traceback.print_exc()
  pass
except BaseException as e:
  print('WARNING: Capstone backend is not available.')
  print(repr(e))
  traceback.print_exc()

if len(available_disassemblers) == 0:
  print('ERROR: No available backend.')
