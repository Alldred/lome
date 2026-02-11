# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Load/store instruction implementations: LB, LH, LW, LBU, LHU, LD, LWU, SB, SH, SW, SD.

Note: Memory model is external and coming soon. These instructions track memory accesses
but do not actually read/write memory yet.
"""

from riscv_model.changes import ChangeRecord, GPRWrite, MemoryAccess
from riscv_model.state import State


def _sign_extend(value: int, bits: int) -> int:
    """Sign extend value from bits to 64 bits."""
    sign_bit = 1 << (bits - 1)
    if value & sign_bit:
        return value | (0xFFFFFFFFFFFFFFFF << bits)
    return value & ((1 << bits) - 1)


def execute_lb(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute LB: rd = sign_extend(mem[rs1 + imm][7:0])"""
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
    """Execute LH: rd = sign_extend(mem[rs1 + imm][15:0])"""
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
    """Execute LW: rd = sign_extend(mem[rs1 + imm][31:0])"""
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
    """Execute LBU: rd = zero_extend(mem[rs1 + imm][7:0])"""
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
    """Execute LHU: rd = zero_extend(mem[rs1 + imm][15:0])"""
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
    """Execute LD: rd = mem[rs1 + imm][63:0]"""
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
    """Execute LWU: rd = zero_extend(mem[rs1 + imm][31:0])"""
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
    """Execute SB: mem[rs1 + imm] = rs2[7:0]"""
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
    """Execute SH: mem[rs1 + imm] = rs2[15:0]"""
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
    """Execute SW: mem[rs1 + imm] = rs2[31:0]"""
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
    """Execute SD: mem[rs1 + imm] = rs2[63:0]"""
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
