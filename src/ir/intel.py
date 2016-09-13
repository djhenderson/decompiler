# -*- coding: utf-8 -*-

""" intel x86 and x64 archs. """

from __future__ import print_function
from collections import namedtuple

from expressions import *
from statements import *

from ir.generic import ir_base

from . import *

# FLAGS
CF =    1 << 0  # carry flag: Set on high-order bit carry or borrow
PF =    1 << 2  # parity flag:
AF =    1 << 4  # adjust flag
ZF =    1 << 6  # zero flag: set if expr == 0
SF =    1 << 7  # sign flag
#~ TF =    1 << 8  # trap flag
#~ IF =    1 << 9  # interrupt enable flag
#~ DF =    1 << 10 # direction flag
OF =    1 << 11 # overflow flag: set when the expression would overflow

# EFLAGS
#~ RF =    1 << 16 # resume flags
#~ VM =    1 << 17 # virtual 8086 mode flag
#~ AC =    1 << 18 # alignment check
#~ VIP =   1 << 19 # virtual interrupt flag
#~ VIF =   1 << 20 # virtual interrupt pending
ID =    1 << 21 # able to use CPUID instruction

SIZE_8 = 8
SIZE_16 = 16
SIZE_32 = 32
SIZE_64 = 64

LOBYTE = 0
HIBYTE = 1
WORD = 2
DWORD = 3
QWORD = 4

register_t = namedtuple('register_t', ['name', 'size', 'type'])
registers = {}

for name in (b'rax', b'rbx', b'rcx', b'rdx', b'rsi', b'rdi', b'rbp', \
    b'rip', b'rsp', b'r8', b'r9', b'r10', b'r11', b'r12', b'r13', \
    b'r14', b'r15'):
  registers[name] = register_t(name, SIZE_64, QWORD)

for name in (b'eax', b'ebx', b'ecx', b'edx', b'esi', b'edi', b'ebp', \
    b'eip', b'esp', b'r8d', b'r9d', b'r10d', b'r11d', b'r12d', \
    b'r13d', b'r14d', b'r15d'):
  registers[name] = register_t(name, SIZE_32, DWORD)

for name in (b'ax', b'bx', b'cx', b'dx', b'si', b'di', b'bp', b'ip', \
    b'sp', b'r8w', b'r9w', b'r10w', b'r11w', b'r12w', b'r13w', \
    b'r14w', b'r15w'):
  registers[name] = register_t(name, SIZE_16, WORD)

for name in (b'ah', b'bh', b'ch', b'dh'):
  registers[name] = register_t(name, SIZE_8, HIBYTE)

for name in (b'al', b'bl', b'cl', b'dl', b'sil', b'dil', b'bpl', \
    b'spl', b'r8b', b'r9b', b'r10b', b'r11b', b'r12b', b'r13b', \
    b'r14b', b'r15b'):
  registers[name] = register_t(name, SIZE_8, LOBYTE)

register_groups = []
register_groups.append((b'rax', b'eax', b'ax', b'ah', b'al'))
register_groups.append((b'rbx', b'ebx', b'bx', b'bh', b'bl'))
register_groups.append((b'rcx', b'ecx', b'cx', b'ch', b'cl'))
register_groups.append((b'rdx', b'edx', b'dx', b'dh', b'dl'))
register_groups.append((b'rsi', b'esi', b'si', b'sil'))
register_groups.append((b'rdi', b'edi', b'di', b'dil'))
register_groups.append((b'rbp', b'ebp', b'bp', b'bpl'))
register_groups.append((b'rip', b'eip', b'ip'))
register_groups.append((b'rsp', b'esp', b'sp', b'spl'))
register_groups.append((b'r8', b'r8d', b'r8w', b'r8b'))
register_groups.append((b'r9', b'r9d', b'r9w', b'r9b'))
register_groups.append((b'r10', b'r10d', b'r10w', b'r10b'))
register_groups.append((b'r11', b'r11d', b'r11w', b'r11b'))
register_groups.append((b'r12', b'r12d', b'r12w', b'r12b'))
register_groups.append((b'r13', b'r13d', b'r13w', b'r13b'))
register_groups.append((b'r14', b'r14d', b'r14w', b'r14b'))
register_groups.append((b'r15', b'r15d', b'r15w', b'r15b'))

## DEBUG
#import pprint
#pp = pprint.PrettyPrinter(indent=3)
#print("registers:")
#pprint.pprint(registers)
## DEBUG


class ir_intel(ir_base):

  def __init__(self):

    assert type(self) != ir_intel, 'must use base classes instead'

    ir_base.__init__(self)

    r = self.get_stack_register()
    self.stackreg = regloc_t(r, self.address_size, name=self.get_regname(r))
    r = self.get_leave_register()
    self.leavereg = regloc_t(r, self.address_size, name=self.get_regname(r))
    r = self.get_result_register()
    self.resultreg = regloc_t(r, self.address_size, name=self.get_regname(r))

    self.special_registers = 9000

    self.eflags_expr = self.make_special_register('%eflags.expr')
    self.cf = self.make_special_register('%eflags.cf')
    self.pf = self.make_special_register('%eflags.pf')
    self.af = self.make_special_register('%eflags.af')
    self.zf = self.make_special_register('%eflags.zf')
    self.sf = self.make_special_register('%eflags.sf')
    self.of = self.make_special_register('%eflags.of')

    self.flow_break = ['retn', 'ret' ] # instructions that break (terminate) the flow
    self.unconditional_jumps = ['jmp', ] # unconditional jumps (one branch)
    self.conditional_jumps = ['jo', 'jno', 'js', 'jns', 'jz', 'je', 'jnz', 'jne',
            'jb', 'jnb', 'jbe', 'ja', 'jl', 'jge', 'jle', 'jg',
            'jpe', 'jno'] # conditional jumps (two branches)

    return

  def get_regindex(self, name):
    if name.lower() in registers:
      return list(registers.keys()).index(name.lower())

  def get_regname(self, which):
    if which < len(registers):
      name = list(registers.keys())[which]
    else:
      name = '#%u' % (which, )
    return name

  def get_stack_register(self):
    if self.ir_id == IR_INTEL_x86:
      return self.get_regindex(b'esp')
    elif self.ir_id == IR_INTEL_x64:
      return self.get_regindex(b'rsp')
    elif self.ir_id == IR_INTEL_x16:
      return self.get_regindex(b'sp')

  def get_result_register(self):
    if self.ir_id == IR_INTEL_x86:
      return self.get_regindex(b'eax')
    elif self.ir_id == IR_INTEL_x64:
      return self.get_regindex(b'rax')
    elif self.ir_id == IR_INTEL_x16:
      return self.get_regindex(b'ax')

  def get_leave_register(self):
    if self.ir_id == IR_INTEL_x86:
      return self.get_regindex(b'ebp')
    elif self.ir_id == IR_INTEL_x64:
      return self.get_regindex(b'rbp')
    elif self.ir_id == IR_INTEL_x16:
      return self.get_regindex(b'bp')

  def make_special_register(self, name):
    reg = flagloc_t(self.special_registers, 1, name)
    self.special_registers += 1
    return reg

  def is_stackreg(self, reg):
    """ return True if the register is the stack register """
    return isinstance(reg, regloc_t) and reg.no_index_eq(self.stackreg)

  def is_stackvar(self, expr):
    return ((type(expr) in (sub_t, add_t) and \
            (self.is_aligned_stackvar(expr.op1) or self.is_stackreg(expr.op1)) \
              and type(expr.op2) == value_t))

  def is_aligned_stackvar(self, expr):
    return type(expr) == and_t and \
            type(expr.op2) == value_t and \
            type(expr.op1) in (sub_t, add_t) and \
            self.is_stackreg(expr.op1.op1) and type(expr.op1.op2) == value_t

  def is_conditional_jump(self, ea):
    """ return true if this instruction is a conditional jump. """
    #print("DEBUG: self.get_mnemonic", repr(self.get_mnemonic))
    mnem = self.get_mnemonic(ea)
    #print("DEBUG: type(mnem):", type(mnem), mnem)
    #print("DEBUG: type(self.conditional_jumps[0]): %s %s" % (type(self.conditional_jumps[0]), self.conditional_jumps[0]))
    if mnem in self.conditional_jumps:
      return True
    return False

  def is_unconditional_jump(self, ea):
    """ return true if this instruction is a unconditional jump. """
    mnem = self.get_mnemonic(ea)
    if mnem in self.unconditional_jumps:
      return True
    return False

  def is_return(self, ea):
    """ return True if this is a return instruction """
    mnem = self.get_mnemonic(ea)
    if mnem in self.flow_break:
      return True
    return False

  def has_jump(self, ea):
    """ return true if this instruction is a jump """
    return self.is_conditional_jump(ea) or self.is_unconditional_jump(ea)

  def next_instruction_ea(self, ea):
    """ return the address of the next instruction. """
    size = self.get_instruction_size(ea)
    assert size > 0, '%x: no instruction' % (ea, )
    return ea + size

  def jump_branches(self, ea):
    mnem = self.get_mnemonic(ea)
    if mnem in self.unconditional_jumps:
      dest = self.get_operand_expression(ea, 0)
      yield dest
    elif mnem in self.conditional_jumps:
      dest = self.get_operand_expression(ea, 0)
      yield dest
      dest = self.next_instruction_ea(ea)
      yield value_t(dest, self.address_size)
    return

  def as_signed(self, v, size=None):
    if size is None:
      size = self.address_size
    if v > (1 << size-1):
      return - ((sum([1 << i for i in range(size)]) + 1) - v)
    return v

  def evaluate_flags(self, expr, flags):
    yield assign_t(self.eflags_expr.copy(), expr.copy())
    if flags & CF:
      yield assign_t(self.cf.copy(), carry_t(self.eflags_expr.copy()))
    if flags & PF:
      yield assign_t(self.pf.copy(), parity_t(self.eflags_expr.copy()))
    if flags & AF:
      yield assign_t(self.af.copy(), adjust_t(self.eflags_expr.copy()))
    if flags & ZF:
      yield assign_t(self.zf.copy(), eq_t(self.eflags_expr.copy(), value_t(0, 1)))
    if flags & SF:
      yield assign_t(self.sf.copy(), sign_t(self.eflags_expr.copy()))
    if flags & OF:
      yield assign_t(self.of.copy(), overflow_t(self.eflags_expr.copy()))
    return

  def set_flags(self, flags, value):
    if flags & CF:
      yield assign_t(self.cf.copy(), value_t(value, 1))
    if flags & PF:
      yield assign_t(self.pf.copy(), value_t(value, 1))
    if flags & AF:
      yield assign_t(self.af.copy(), value_t(value, 1))
    if flags & ZF:
      yield assign_t(self.zf.copy(), value_t(value, 1))
    if flags & SF:
      yield assign_t(self.sf.copy(), value_t(value, 1))
    if flags & OF:
      yield assign_t(self.of.copy(), value_t(value, 1))
    return

  def generate_statements(self, ea):

    mnem = self.get_mnemonic(ea)

    expr = None

    if mnem in ('nop', 'hlt'):
      pass

    elif mnem in ('cdq', 'cdqe'):
      # sign extension... not supported until we do type analysis
      pass

    elif mnem == 'push':

      op = self.get_operand_expression(ea, 0)

      # stack location assignment
      expr = assign_t(deref_t(self.stackreg.copy(), self.address_size), op.copy())
      yield expr

      # stack pointer modification
      expr = assign_t(self.stackreg.copy(), sub_t(self.stackreg.copy(), value_t(4, self.address_size)))
      yield expr

    elif mnem == 'pop':
      #~ assert insn.Op1.type == 1

      # stack pointer modification
      expr = assign_t(self.stackreg.copy(), add_t(self.stackreg.copy(), value_t(4, self.address_size)))
      yield expr

      # stack location value
      dst = self.get_operand_expression(ea, 0)

      expr = assign_t(dst.copy(), deref_t(self.stackreg.copy(), self.address_size))
      yield expr

    elif mnem == 'leave':

      # mov esp, ebp
      expr = assign_t(self.stackreg.copy(), self.leavereg.copy())
      yield expr

      # stack pointer modification
      expr = assign_t(self.stackreg.copy(), add_t(self.stackreg.copy(), value_t(4, self.address_size)))
      yield expr

      # stack location value
      expr = assign_t(self.leavereg.copy(), deref_t(self.stackreg.copy(), self.address_size))
      yield expr

    elif mnem == 'call':
      # call is a special case: we analyse the target functions's flow to determine
      # the likely parameters.

      expr, spoils = self.get_call_expression(ea)
      yield expr

    elif mnem == 'lea':
      #~ assert insn.Op1.type == 1

      dst = self.get_operand_expression(ea, 0)
      op = self.get_operand_expression(ea, 1)

      expr = assign_t(dst, address_t(op))
      yield expr

    elif mnem == 'not':

      op = self.get_operand_expression(ea, 0)

      expr = assign_t(op.copy(), not_t(op))
      yield expr

    elif mnem == 'neg':

      op = self.get_operand_expression(ea, 0)

      expr = assign_t(op.copy(), neg_t(op))
      yield expr

    elif mnem in ('mov', 'movzx', 'movsxd', 'movsx'):

      dst = self.get_operand_expression(ea, 0)
      op = self.get_operand_expression(ea, 1)

      expr = assign_t(dst, op)
      yield expr

    elif mnem in ('inc', 'dec'):
      choices = {'inc': add_t, 'dec': sub_t}

      op1 = self.get_operand_expression(ea, 0)
      op2 = value_t(1, self.address_size)

      expr = (choices[mnem])(op1, op2)

      # CF is unaffected
      for _expr in self.evaluate_flags(expr, PF | AF | ZF | SF | OF):
        yield _expr

      yield assign_t(op1.copy(), expr)

    elif mnem in ('add', 'sub'):
      choices = {'add': add_t, 'sub': sub_t}

      op1 = self.get_operand_expression(ea, 0)
      op2 = self.get_operand_expression(ea, 1)

      expr = (choices[mnem])(op1, op2)

      for _expr in self.evaluate_flags(expr, CF | PF | AF | ZF | SF | OF):
        yield _expr

      yield assign_t(op1.copy(), expr)

    elif mnem in ('imul', ):
      choices = {'imul': mul_t, }

      op1 = self.get_operand_expression(ea, 0)
      op2 = self.get_operand_expression(ea, 1)

      expr = (choices[mnem])(op1, op2)

      #~ # TODO: SF, ZF, AF, PF is undefined
      #~ # TODO: CF, OF is defined..

      yield assign_t(op1.copy(), expr)

    elif mnem in ('xor', 'or', 'and'):
      choices = {'xor': xor_t, 'or': or_t, 'and': and_t}

      op1 = self.get_operand_expression(ea, 0)
      op2 = self.get_operand_expression(ea, 1)

      expr = (choices[mnem])(op1, op2)

      for _expr in self.set_flags(CF | OF, value=0):
        yield _expr
      # TODO: AF is undefined
      for _expr in self.evaluate_flags(expr, PF | ZF | SF):
        yield _expr

      yield assign_t(op1.copy(), expr)

    elif mnem in ('shl', 'shr', 'sal', 'sar'):
      choices = {'shr': shr_t, 'shl': shl_t, 'sar': shr_t, 'sal': shl_t}

      op1 = self.get_operand_expression(ea, 0)
      op2 = self.get_operand_expression(ea, 1)

      expr = (choices[mnem])(op1, op2)

      for _expr in self.evaluate_flags(expr, CF | PF | AF | ZF | SF | OF):
        yield _expr

      yield assign_t(op1.copy(), expr)

    elif mnem in ('retn', 'ret'):
      #~ assert insn.Op1.type in (0, 5)

      #~ if insn.Op1.type == 5:
        #~ # stack pointer adjusted from return
        #~ op = self.get_operand(ea, insn.Op1)
        #~ expr = assign_t(self.stackreg.copy(), add_t(self.stackreg.copy(), op))
        #~ yield expr

      expr = return_t(ea, self.resultreg.copy())
      yield expr

    elif mnem == 'cmp':
      # The comparison is performed by subtracting the second operand from
      # the first operand and then setting the status flags in the same manner
      # as the SUB instruction.

      op1 = self.get_operand_expression(ea, 0)
      op2 = self.get_operand_expression(ea, 1)

      for expr in self.evaluate_flags(sub_t(op1, op2), CF | PF | AF | ZF | SF | OF):
        yield expr

    elif mnem == 'test':

      op1 = self.get_operand_expression(ea, 0)
      op2 = self.get_operand_expression(ea, 1)

      for expr in self.set_flags(CF | OF, value=0):
        yield expr

      # TODO: AF is undefined..

      for expr in self.evaluate_flags(and_t(op1, op2), PF | ZF | SF):
        yield expr

    elif mnem == 'jmp':
      # control flow instruction...

      dst = self.get_operand_expression(ea, 0)

      if type(dst) == value_t and self.get_function_start(dst.value) == dst.value:
        # target of jump is a function.
        # let's assume that this is tail call optimization.

        expr = return_t(ea, call_t(dst, self.resultreg.copy(), params_t()))
        yield expr

        #~ block.return_expr = expr
      else:
        expr = goto_t(ea, dst)
        yield expr

    elif mnem in ('cmova', 'cmovae', 'cmovb', 'cmovbe', 'cmovc', 'cmove', 'cmovg',
                    'cmovge', 'cmovl', 'cmovle', 'cmovna', 'cmovnae', 'cmovbe',
                    'cmovnc', 'cmovne', 'cmovng', 'cmovnge', 'cmovnl', 'cmovnle',
                    'cmovno', 'cmovnp', 'cmovns', 'cmovnz', 'cmovo', 'cmovp',
                    'cmovpe', 'cmovpo', 'cmovs', 'cmovz'):
      # CMOVcc (conditional mov)

      op1 = self.get_operand_expression(ea, 0)
      op2 = self.get_operand_expression(ea, 1)

      if mnem == 'cmova':
        cond = b_and_t(b_not_t(self.zf.copy()), b_not_t(self.cf.copy()))
      elif mnem in ('cmovae', 'cmovnb', 'cmovnc'):
        cond = b_not_t(self.cf.copy())
      elif mnem in ('cmov', 'cmovc', 'cmovnae'):
        cond = self.cf.copy()
      elif mnem == 'cmovbe':
        cond = b_or_t(self.zf.copy(), self.cf.copy())
      elif mnem == 'cmove':
        cond = self.zf.copy()
      elif mnem in ('cmovg', 'cmovnle'):
        cond = b_and_t(b_not_t(self.zf.copy()), eq_t(self.sf.copy(), self.of.copy()))
      elif mnem in ('cmovge', 'cmovnl'):
        cond = eq_t(self.sf.copy(), self.of.copy())
      elif mnem in ('cmovl', 'cmovnge'):
        cond = neq_t(self.sf.copy(), self.of.copy())
      elif mnem in ('cmovle', 'cmovng'):
        cond = b_or_t(self.zf.copy(), neq_t(self.sf.copy(), self.of.copy()))
      elif mnem == 'cmovna':
        cond = b_or_t(self.zf.copy(), self.cf.copy(), )
      elif mnem == 'cmovnbe':
        cond = b_and_t(b_not_t(self.zf.copy()), b_not_t(self.cf.copy()))
      elif mnem in ('cmovnz', 'cmovne'):
        cond = b_not_t(self.zf.copy())
      elif mnem in ('cmovno', ):
        cond = b_not_t(self.of.copy())
      elif mnem in ('cmovnp', 'cmovpo'):
        cond = b_not_t(self.pf.copy())
      elif mnem in ('cmovns', ):
        cond = b_not_t(self.sf.copy())
      elif mnem in ('cmovo', ):
        cond = self.of.copy()
      elif mnem in ('cmovo', ):
        cond = self.of.copy()
      elif mnem in ('cmovp', 'cmovpe'):
        cond = self.pf.copy()
      elif mnem in ('cmovs', ):
        cond = self.sf.copy()
      elif mnem in ('cmovz', ):
        cond = self.zf.copy()

      expr = assign_t(op1.copy(), ternary_if_t(cond, op2, op1))
      yield expr

    elif mnem in ('seta', 'setae', 'setb', 'setbe', 'setc', 'sete', 'setg',
                    'setge', 'setl', 'setle', 'setna', 'setnae', 'setbe',
                    'setnc', 'setne', 'setng', 'setnge', 'setnl', 'setnle',
                    'setno', 'setnp', 'setns', 'setnz', 'seto', 'setp',
                    'setpe', 'setpo', 'sets', 'setz'):

      op1 = self.get_operand_expression(ea, 0)

      # http://faydoc.tripod.com/cpu/setnz.htm
      if mnem == 'seta':
        cond = b_and_t(b_not_t(self.zf.copy()), b_not_t(self.cf.copy()))
      elif mnem in ('setae', 'setnb', 'setnc'):
        cond = b_not_t(self.cf.copy())
      elif mnem in ('set', 'setc', 'setnae'):
        cond = self.cf.copy()
      elif mnem == 'setbe':
        cond = b_or_t(self.zf.copy(), self.cf.copy())
      elif mnem == 'sete':
        cond = self.zf.copy()
      elif mnem in ('setg', 'setnle'):
        cond = b_and_t(b_not_t(self.zf.copy()), eq_t(self.sf.copy(), self.of.copy()))
      elif mnem in ('setge', 'setnl'):
        cond = eq_t(self.sf.copy(), self.of.copy())
      elif mnem in ('setl', 'setnge'):
        cond = neq_t(self.sf.copy(), self.of.copy())
      elif mnem in ('setle', 'setng'):
        cond = b_or_t(self.zf.copy(), neq_t(self.sf.copy(), self.of.copy()))
      elif mnem == 'setna':
        cond = b_or_t(self.zf.copy(), self.cf.copy(), )
      elif mnem == 'setnbe':
        cond = b_and_t(b_not_t(self.zf.copy()), b_not_t(self.cf.copy()))
      elif mnem in ('setnz', 'setne'):
        cond = b_not_t(self.zf.copy())
      elif mnem in ('setno', ):
        cond = b_not_t(self.of.copy())
      elif mnem in ('setnp', 'setpo'):
        cond = b_not_t(self.pf.copy())
      elif mnem in ('setns', ):
        cond = b_not_t(self.sf.copy())
      elif mnem in ('seto', ):
        cond = self.of.copy()
      elif mnem in ('seto', ):
        cond = self.of.copy()
      elif mnem in ('setp', 'setpe'):
        cond = self.pf.copy()
      elif mnem in ('sets', ):
        cond = self.sf.copy()
      elif mnem in ('setz', ):
        cond = self.zf.copy()

      expr = assign_t(op1, cond)
      yield expr

    elif mnem in self.conditional_jumps:
      # we do not distinguish between signed and unsigned comparision here.

      if mnem == 'jns':
        # jump if sign bit is clear
        cond = b_not_t(self.sf.copy())
      elif mnem == 'js':
        # jump if sign bit is set
        cond = self.sf.copy()
      elif mnem in ('jnz', 'jne'):
        # jump if zero bit is clear
        cond = b_not_t(self.zf.copy())
      elif mnem in ('jz', 'je'):
        # jump if zero bit is set
        cond = self.zf.copy()
      elif mnem == 'jno':
        # jump if overflow bit is clear
        cond = b_not_t(self.of.copy())
      elif mnem == 'jo':
        # jump if overflow bit is set
        cond = self.of.copy()
      elif mnem == 'jnb': # jae jnc
        # jump if carry bit is clear
        cond = b_not_t(self.cf.copy())
      elif mnem == 'jb': # jnae jc
        # jump if carry bit is set
        cond = self.cf.copy()
      elif mnem == 'jbe': # jna
        # jump if below or equal
        cond = b_or_t(self.zf.copy(), self.cf.copy())
      elif mnem == 'ja': # jnbe
        # jump if above
        cond = b_and_t(b_not_t(self.zf.copy()), b_not_t(self.cf.copy()))
      elif mnem == 'jl': # jnge
        # jump if less
        cond = neq_t(self.sf.copy(), self.of.copy())
      elif mnem == 'jge': # jnl
        # jump if greater or equal
        cond = eq_t(self.sf.copy(), self.of.copy())
      elif mnem == 'jle': # jng
        # jump if less or equal
        cond = b_or_t(self.zf.copy(), neq_t(self.sf.copy(), self.of.copy()))
      elif mnem == 'jg': # jnle
        # jump if greater
        cond = b_and_t(b_not_t(self.zf.copy()), eq_t(self.sf.copy(), self.of.copy()))
      elif mnem == 'jpe': # jp
        # jump if parity even
        cond = self.pf.copy()
      elif mnem == 'jpo': # jnp
        # jump if parity odd
        cond = b_not_t(self.pf.copy())
      else:
        raise RuntimeError('unknown jump mnemonic')

      true = self.get_operand_expression(ea, 0)
      false = value_t(self.next_instruction_ea(ea), self.address_size)

      expr = branch_t(ea, cond, true, false)
      yield expr
    else:
      raise RuntimeError('%x: not yet handled instruction: %s ' % (ea, mnem))

    return


class ir_intel_x86(ir_intel):

  def __init__(self):
    self.address_size = 32
    ir_intel.__init__(self)
    return

  def get_register_size(self, which):
    return 32


class ir_intel_x64(ir_intel):

  def __init__(self):
    self.address_size = 64
    ir_intel.__init__(self)
    return

  def get_register_size(self, which):
    return 64


class ir_intel_x16(ir_intel):

  def __init__(self):
    self.address_size = 20
    ir_intel.__init__(self)
    return

  def get_register_size(self, which):
    return 16
