# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.insert(0, './tests/')
sys.path.insert(0, './src/')

import capstone as cs
import decompiler
from host import dis
from output import c
from djh_x86 import print_insn_detail

print("INFO: Capstone core version: %d.%d.%d" % cs.cs_version())
print("INFO: Capstone python binding version: %d.%d.%d" % cs.version_bind())

# Create a Capstone object, which will be used as disassembler
md = cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_32)

# Define a bunch of bytes to disassemble (Capstone requires bytes)
code = \
    b"\x55\x89\xe5\x83\xec\x28\xc7\x45\xf4\x00\x00\x00\x00\x8b\x45\xf4" + \
    b"\x8b\x00\x83\xf8\x0e\x75\x0c\xc7\x04\x24\x30\x87\x04\x08\xe8\xd3" + \
    b"\xfe\xff\xff\xb8\x00\x00\x00\x00\xc9\xc3"

if True:
    try:
        #md = Cs(arch, mode)
        md.detail = True

        for insn in md.disasm(code, 0x1000):
            print_insn_detail(cs.CS_MODE_32, insn)
        print ("0x%x:\n" % (insn.address + insn.size))
    except cs.CsError as e:
        print("ERROR: %s" % e)

# Create the capstone-specific backend; it will yield expressions that the decompiler is able to use.
disasm = dis.available_disassemblers['capstone'].create(md, code, 0x1000)

# Create the decompiler
dec = decompiler.decompiler_t(disasm, 0x1000)

# Transform the function until it is decompiled
dec.step_until(decompiler.step_decompiled)

# Tokenize and output the function as string
print(''.join([str(o) for o in c.tokenizer(dec.function).tokens]))

print("expect:", """
func() {
   s0 = 0;
   if (*s0 == 14) {
      s2 = 134514480;
      3830();
   }
   return 0;
}""")
