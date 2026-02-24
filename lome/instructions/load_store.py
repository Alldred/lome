# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Load/store instruction implementations: LB, LH, LW, LBU, LHU, LD, LWU, SB, SH, SW, SD.

When a :class:`~lome.memory.MemoryInterface` is provided, loads and stores
read/write real memory. Otherwise loads return 0 (placeholder for tests).
"""

from __future__ import annotations

from typing import Optional

from lome.changes import ChangeRecord, GPRRead, GPRWrite, MemoryAccess
from lome.memory import MemoryInterface
from lome.state import State


def _sign_extend(value: int, bits: int) -> int:
    """Sign extend *value* from *bits* width to 64 bits.

    Parameters:
        value: The narrow value to extend.
        bits: Original bit-width of *value* (e.g. 8, 16, 32).

    Returns:
        The 64-bit sign-extended integer.
    """
    sign_bit = 1 << (bits - 1)
    if value & sign_bit:
        return value | (0xFFFFFFFFFFFFFFFF << bits)
    return value & ((1 << bits) - 1)


def execute_lb(
    operand_values: dict,
    state: State,
    pc: int,
    *,
    memory: Optional[MemoryInterface] = None,
) -> ChangeRecord:
    """Execute LB: rd = sign_extend(mem[rs1 + imm][7:0])

    Load a byte from memory at address *rs1 + imm*, sign-extend it to
    64 bits, and write the result to rd.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the memory access and GPR write.

    Example::

        # lb x1, 0(x2)  — load sign-extended byte from address in x2
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    addr = rs1_val + imm

    value = memory.load(addr, 1) if memory else 0
    result = _sign_extend(value, 8)

    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=1, is_write=False)
    )
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_lh(
    operand_values: dict,
    state: State,
    pc: int,
    *,
    memory: Optional[MemoryInterface] = None,
) -> ChangeRecord:
    """Execute LH: rd = sign_extend(mem[rs1 + imm][15:0])

    Load a halfword (2 bytes) from memory at address *rs1 + imm*,
    sign-extend it to 64 bits, and write the result to rd.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the memory access and GPR write.

    Example::

        # lh x1, 4(x2)  — load sign-extended halfword from x2+4
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    addr = rs1_val + imm

    value = memory.load(addr, 2) if memory else 0
    result = _sign_extend(value, 16)

    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=2, is_write=False)
    )
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_lw(
    operand_values: dict,
    state: State,
    pc: int,
    *,
    memory: Optional[MemoryInterface] = None,
) -> ChangeRecord:
    """Execute LW: rd = sign_extend(mem[rs1 + imm][31:0])

    Load a word (4 bytes) from memory at address *rs1 + imm*, sign-extend
    it to 64 bits, and write the result to rd.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the memory access and GPR write.

    Example::

        # lw x1, 8(x2)  — load sign-extended word from x2+8
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    addr = rs1_val + imm

    value = memory.load(addr, 4) if memory else 0
    result = _sign_extend(value, 32)

    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=4, is_write=False)
    )
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_lbu(
    operand_values: dict,
    state: State,
    pc: int,
    *,
    memory: Optional[MemoryInterface] = None,
) -> ChangeRecord:
    """Execute LBU: rd = zero_extend(mem[rs1 + imm][7:0])

    Load a byte from memory at address *rs1 + imm*, zero-extend it to
    64 bits, and write the result to rd.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the memory access and GPR write.

    Example::

        # lbu x1, 0(x2)  — load zero-extended byte from address in x2
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    addr = rs1_val + imm

    value = memory.load(addr, 1) if memory else 0
    result = value & 0xFF  # Zero extend

    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=1, is_write=False)
    )
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_lhu(
    operand_values: dict,
    state: State,
    pc: int,
    *,
    memory: Optional[MemoryInterface] = None,
) -> ChangeRecord:
    """Execute LHU: rd = zero_extend(mem[rs1 + imm][15:0])

    Load a halfword (2 bytes) from memory at address *rs1 + imm*,
    zero-extend it to 64 bits, and write the result to rd.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the memory access and GPR write.

    Example::

        # lhu x1, 2(x2)  — load zero-extended halfword from x2+2
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    addr = rs1_val + imm

    value = memory.load(addr, 2) if memory else 0
    result = value & 0xFFFF  # Zero extend

    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=2, is_write=False)
    )
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_ld(
    operand_values: dict,
    state: State,
    pc: int,
    *,
    memory: Optional[MemoryInterface] = None,
) -> ChangeRecord:
    """Execute LD: rd = mem[rs1 + imm][63:0]

    Load a doubleword (8 bytes) from memory at address *rs1 + imm* and
    write the result to rd.  RV64 only.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the memory access and GPR write.

    Example::

        # ld x1, 16(x2)  — load doubleword from x2+16
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    addr = rs1_val + imm

    result = memory.load(addr, 8) if memory else 0

    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=result, size=8, is_write=False)
    )
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_lwu(
    operand_values: dict,
    state: State,
    pc: int,
    *,
    memory: Optional[MemoryInterface] = None,
) -> ChangeRecord:
    """Execute LWU: rd = zero_extend(mem[rs1 + imm][31:0])

    Load a word (4 bytes) from memory at address *rs1 + imm*, zero-extend
    it to 64 bits, and write the result to rd.  RV64 only.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the memory access and GPR write.

    Example::

        # lwu x1, 4(x2)  — load zero-extended word from x2+4
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    addr = rs1_val + imm

    value = memory.load(addr, 4) if memory else 0
    result = value & 0xFFFFFFFF  # Zero extend

    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=4, is_write=False)
    )
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_sb(
    operand_values: dict,
    state: State,
    pc: int,
    *,
    memory: Optional[MemoryInterface] = None,
) -> ChangeRecord:
    """Execute SB: mem[rs1 + imm] = rs2[7:0]

    Store the lowest byte of rs2 to memory at address *rs1 + imm*.

    Parameters:
        operand_values: dict with keys ``rs1``, ``rs2``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the memory access.

    Example::

        # sb x3, 0(x2)  — store low byte of x3 to address in x2
    """
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    addr = rs1_val + imm
    value = rs2_val & 0xFF

    if memory is not None:
        memory.store(addr, value, 1)
    changes = ChangeRecord()
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    if rs2_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs2_idx, value=rs2_val))
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=1, is_write=True)
    )
    return changes


def execute_sh(
    operand_values: dict,
    state: State,
    pc: int,
    *,
    memory: Optional[MemoryInterface] = None,
) -> ChangeRecord:
    """Execute SH: mem[rs1 + imm] = rs2[15:0]

    Store the lowest halfword (2 bytes) of rs2 to memory at address
    *rs1 + imm*.

    Parameters:
        operand_values: dict with keys ``rs1``, ``rs2``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the memory access.

    Example::

        # sh x3, 2(x2)  — store low halfword of x3 to x2+2
    """
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    addr = rs1_val + imm
    value = rs2_val & 0xFFFF

    if memory is not None:
        memory.store(addr, value, 2)
    changes = ChangeRecord()
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    if rs2_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs2_idx, value=rs2_val))
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=2, is_write=True)
    )
    return changes


def execute_sw(
    operand_values: dict,
    state: State,
    pc: int,
    *,
    memory: Optional[MemoryInterface] = None,
) -> ChangeRecord:
    """Execute SW: mem[rs1 + imm] = rs2[31:0]

    Store the lowest word (4 bytes) of rs2 to memory at address
    *rs1 + imm*.

    Parameters:
        operand_values: dict with keys ``rs1``, ``rs2``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the memory access.

    Example::

        # sw x3, 8(x2)  — store low word of x3 to x2+8
    """
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    addr = rs1_val + imm
    value = rs2_val & 0xFFFFFFFF

    if memory is not None:
        memory.store(addr, value, 4)
    changes = ChangeRecord()
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    if rs2_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs2_idx, value=rs2_val))
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=4, is_write=True)
    )
    return changes


def execute_sd(
    operand_values: dict,
    state: State,
    pc: int,
    *,
    memory: Optional[MemoryInterface] = None,
) -> ChangeRecord:
    """Execute SD: mem[rs1 + imm] = rs2[63:0]

    Store the full doubleword (8 bytes) of rs2 to memory at address
    *rs1 + imm*.  RV64 only.

    Parameters:
        operand_values: dict with keys ``rs1``, ``rs2``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the memory access.

    Example::

        # sd x3, 16(x2)  — store doubleword x3 to x2+16
    """
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    addr = rs1_val + imm
    value = rs2_val

    if memory is not None:
        memory.store(addr, value, 8)

    changes = ChangeRecord()
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    if rs2_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs2_idx, value=rs2_val))
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=8, is_write=True)
    )
    return changes
