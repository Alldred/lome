# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Load/store instruction implementations: LB, LH, LW, LBU, LHU, LD, LWU, SB, SH, SW, SD.

Note: Memory model is external and coming soon. These instructions track memory accesses
but do not actually read/write memory yet.
"""

from __future__ import annotations

from riscv_model.changes import ChangeRecord, GPRWrite, MemoryAccess
from riscv_model.state import State


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


def execute_lb(operand_values: dict, state: State, pc: int) -> ChangeRecord:
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
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    addr = rs1_val + imm

    # Memory model not yet implemented - track access
    # In real implementation, would read byte from memory
    value = 0  # Placeholder
    result = _sign_extend(value, 8)

    changes = ChangeRecord()
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=None, size=1, is_write=False)
    )
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_lh(operand_values: dict, state: State, pc: int) -> ChangeRecord:
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
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    addr = rs1_val + imm

    # Memory model not yet implemented - track access
    value = 0  # Placeholder
    result = _sign_extend(value, 16)

    changes = ChangeRecord()
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=None, size=2, is_write=False)
    )
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_lw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
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
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    addr = rs1_val + imm

    # Memory model not yet implemented - track access
    value = 0  # Placeholder
    result = _sign_extend(value, 32)

    changes = ChangeRecord()
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=None, size=4, is_write=False)
    )
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_lbu(operand_values: dict, state: State, pc: int) -> ChangeRecord:
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
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    addr = rs1_val + imm

    # Memory model not yet implemented - track access
    value = 0  # Placeholder
    result = value & 0xFF  # Zero extend

    changes = ChangeRecord()
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=None, size=1, is_write=False)
    )
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_lhu(operand_values: dict, state: State, pc: int) -> ChangeRecord:
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
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    addr = rs1_val + imm

    # Memory model not yet implemented - track access
    value = 0  # Placeholder
    result = value & 0xFFFF  # Zero extend

    changes = ChangeRecord()
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=None, size=2, is_write=False)
    )
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_ld(operand_values: dict, state: State, pc: int) -> ChangeRecord:
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
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    addr = rs1_val + imm

    # Memory model not yet implemented - track access
    result = 0  # Placeholder

    changes = ChangeRecord()
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=None, size=8, is_write=False)
    )
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_lwu(operand_values: dict, state: State, pc: int) -> ChangeRecord:
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
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    addr = rs1_val + imm

    # Memory model not yet implemented - track access
    value = 0  # Placeholder
    result = value & 0xFFFFFFFF  # Zero extend

    changes = ChangeRecord()
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=None, size=4, is_write=False)
    )
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_sb(operand_values: dict, state: State, pc: int) -> ChangeRecord:
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

    # Memory model not yet implemented - track access
    changes = ChangeRecord()
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=1, is_write=True)
    )
    return changes


def execute_sh(operand_values: dict, state: State, pc: int) -> ChangeRecord:
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

    # Memory model not yet implemented - track access
    changes = ChangeRecord()
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=2, is_write=True)
    )
    return changes


def execute_sw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
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

    # Memory model not yet implemented - track access
    changes = ChangeRecord()
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=4, is_write=True)
    )
    return changes


def execute_sd(operand_values: dict, state: State, pc: int) -> ChangeRecord:
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

    # Memory model not yet implemented - track access
    changes = ChangeRecord()
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=8, is_write=True)
    )
    return changes
