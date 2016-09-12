# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import capstone

import ir
import ir.intel

from . import intel

def disassembler_for_arch(md):

  if md.arch == capstone.CS_ARCH_X86 and md.mode & capstone.CS_MODE_32:
    print('Architecture: 32-bit intel.')
    return (ir.IR_INTEL_x86, ir.intel.ir_intel_x86, intel.disassembler)
  elif md.arch == capstone.CS_ARCH_X86 and md.mode & capstone.CS_MODE_64:
    print('Architecture: 64-bit intel.')
    return (ir.IR_INTEL_x64, ir.intel.ir_intel_x64, intel.disassembler)

  raise RuntimeError("Don't know which arch to choose for %s" % (repr(filetype), ))

def create(md, code, ea=0):
  """
  Return a new instance of a disassembler made up of the generic
  architecture support (from ir/*.py) and the specific host disassembler
  for this architecture.
  """

  if sys.version_info[0] >= 3:
     # version 3+, binary code must be bytes, not str
     if not isinstance(code, (bytes, bytearray)):
        raise TypeError("Binary code must be bytes-like, not e.g. str")
  else:
     # versions < 3, binary code must be str, not unicode
     if not isinstance(code, str):
        raise TypeError("Binary code must be string, not e.g. unicode")

  ir_id, ir_cls, dis_cls = disassembler_for_arch(md)

  class disassembler(dis_cls, ir_cls): # disassembler (host) class must be left-most.

    def __init__(self, ir_id, md, code, ea):
      self.ir_id = ir_id
      self.md = md
      self.code = code
      self.ea = ea
      dis_cls.__init__(self)
      ir_cls.__init__(self)
      return

  dis = disassembler(ir_id, md, code, ea)

  return dis
