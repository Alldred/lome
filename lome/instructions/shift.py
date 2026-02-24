# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Shift instruction implementations: SLL, SLLI, SRL, SRLI, SRA, SRAI, SLLW, SLLIW, SRLW, SRLIW, SRAW, SRAIW."""

from __future__ import annotations

from lome.changes import ChangeRecord, GPRRead, GPRWrite
from lome.state import State


def execute_sll(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SLL: rd = rs1 << (rs2 & 0x3F)

    Logical left shift rs1 by the amount in the lower 6 bits of rs2,
    storing the 64-bit result in rd.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``rs2``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # sll x1, x2, x3  — x1 = x2 << (x3 & 0x3F)
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    if rs2_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs2_idx, value=rs2_val))
    shamt = rs2_val & 0x3F  # Lower 6 bits
    result = (rs1_val << shamt) & 0xFFFFFFFFFFFFFFFF

    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_slli(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SLLI: rd = rs1 << (imm & 0x3F)

    Logical left shift rs1 by the immediate shift amount (lower 6 bits),
    storing the 64-bit result in rd.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # slli x1, x2, 4  — x1 = x2 << 4
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    shamt = imm & 0x3F  # Lower 6 bits
    result = (rs1_val << shamt) & 0xFFFFFFFFFFFFFFFF

    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_srl(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SRL: rd = rs1 >> (rs2 & 0x3F) (logical right shift)

    Logical right shift rs1 by the amount in the lower 6 bits of rs2,
    zero-filling vacated upper bits.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``rs2``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # srl x1, x2, x3  — x1 = x2 >> (x3 & 0x3F) (logical)
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    if rs2_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs2_idx, value=rs2_val))
    shamt = rs2_val & 0x3F  # Lower 6 bits
    result = (rs1_val >> shamt) & 0xFFFFFFFFFFFFFFFF

    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_srli(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SRLI: rd = rs1 >> (imm & 0x3F) (logical right shift)

    Logical right shift rs1 by the immediate shift amount (lower 6 bits),
    zero-filling vacated upper bits.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # srli x1, x2, 8  — x1 = x2 >> 8 (logical)
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    shamt = imm & 0x3F  # Lower 6 bits
    result = (rs1_val >> shamt) & 0xFFFFFFFFFFFFFFFF

    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_sra(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SRA: rd = rs1 >> (rs2 & 0x3F) (arithmetic right shift)

    Arithmetic right shift rs1 by the amount in the lower 6 bits of rs2,
    sign-filling vacated upper bits.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``rs2``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # sra x1, x2, x3  — x1 = x2 >> (x3 & 0x3F) (arithmetic)
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    if rs2_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs2_idx, value=rs2_val))
    shamt = rs2_val & 0x3F  # Lower 6 bits
    # Arithmetic right shift: sign extend
    if rs1_val & 0x8000000000000000:
        # Negative number, preserve sign
        mask = (1 << (64 - shamt)) - 1
        mask = ~mask
        result = ((rs1_val >> shamt) | mask) & 0xFFFFFFFFFFFFFFFF
    else:
        result = (rs1_val >> shamt) & 0xFFFFFFFFFFFFFFFF

    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_srai(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SRAI: rd = rs1 >> (imm & 0x3F) (arithmetic right shift)

    Arithmetic right shift rs1 by the immediate shift amount (lower
    6 bits), sign-filling vacated upper bits.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # srai x1, x2, 4  — x1 = x2 >> 4 (arithmetic)
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    shamt = imm & 0x3F  # Lower 6 bits
    # Arithmetic right shift: sign extend
    if rs1_val & 0x8000000000000000:
        # Negative number, preserve sign
        mask = (1 << (64 - shamt)) - 1
        mask = ~mask
        result = ((rs1_val >> shamt) | mask) & 0xFFFFFFFFFFFFFFFF
    else:
        result = (rs1_val >> shamt) & 0xFFFFFFFFFFFFFFFF

    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_sllw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SLLW: rd = sign_extend((rs1 << (rs2 & 0x1F))[31:0])

    Left shift the lower 32 bits of rs1 by the amount in the lower
    5 bits of rs2, sign-extend the 32-bit result to 64 bits.  RV64 only.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``rs2``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # sllw x1, x2, x3  — x1 = sext32(x2[31:0] << (x3 & 0x1F))
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    rs2_val = state.get_gpr(rs2_idx)
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    if rs2_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs2_idx, value=rs2_val))
    shamt = rs2_val & 0x1F  # Lower 5 bits for 32-bit shift
    result_32 = (rs1_val << shamt) & 0xFFFFFFFF
    # Sign extend from 32 bits
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32

    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_slliw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SLLIW: rd = sign_extend((rs1 << (imm & 0x1F))[31:0])

    Left shift the lower 32 bits of rs1 by the immediate (lower 5 bits),
    sign-extend the 32-bit result to 64 bits.  RV64 only.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # slliw x1, x2, 3  — x1 = sext32(x2[31:0] << 3)
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    shamt = imm & 0x1F  # Lower 5 bits for 32-bit shift
    result_32 = (rs1_val << shamt) & 0xFFFFFFFF
    # Sign extend from 32 bits
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32

    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_srlw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SRLW: rd = sign_extend((rs1 >> (rs2 & 0x1F))[31:0]) (logical)

    Logical right shift the lower 32 bits of rs1 by the amount in the
    lower 5 bits of rs2, sign-extend the 32-bit result to 64 bits.
    RV64 only.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``rs2``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # srlw x1, x2, x3  — x1 = sext32(x2[31:0] >> (x3 & 0x1F))
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    rs2_val = state.get_gpr(rs2_idx)
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    if rs2_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs2_idx, value=rs2_val))
    shamt = rs2_val & 0x1F  # Lower 5 bits for 32-bit shift
    result_32 = (rs1_val >> shamt) & 0xFFFFFFFF
    # Sign extend from 32 bits
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32

    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_srliw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SRLIW: rd = sign_extend((rs1 >> (imm & 0x1F))[31:0]) (logical)

    Logical right shift the lower 32 bits of rs1 by the immediate (lower
    5 bits), sign-extend the 32-bit result to 64 bits.  RV64 only.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # srliw x1, x2, 8  — x1 = sext32(x2[31:0] >> 8) (logical)
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    shamt = imm & 0x1F  # Lower 5 bits for 32-bit shift
    result_32 = (rs1_val >> shamt) & 0xFFFFFFFF
    # Sign extend from 32 bits
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32

    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_sraw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SRAW: rd = sign_extend((rs1 >> (rs2 & 0x1F))[31:0]) (arithmetic)

    Arithmetic right shift the lower 32 bits of rs1 by the amount in the
    lower 5 bits of rs2, sign-extend the 32-bit result to 64 bits.
    RV64 only.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``rs2``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # sraw x1, x2, x3  — x1 = sext32(x2[31:0] >>a (x3 & 0x1F))
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    rs2_val = state.get_gpr(rs2_idx)
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    if rs2_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs2_idx, value=rs2_val))
    shamt = rs2_val & 0x1F  # Lower 5 bits for 32-bit shift
    # Arithmetic right shift on 32-bit value
    if rs1_val & 0x80000000:
        # Negative number, preserve sign
        mask = (1 << (32 - shamt)) - 1
        mask = ~mask
        result_32 = ((rs1_val >> shamt) | mask) & 0xFFFFFFFF
    else:
        result_32 = (rs1_val >> shamt) & 0xFFFFFFFF
    # Sign extend from 32 bits
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32

    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_sraiw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SRAIW: rd = sign_extend((rs1 >> (imm & 0x1F))[31:0]) (arithmetic)

    Arithmetic right shift the lower 32 bits of rs1 by the immediate
    (lower 5 bits), sign-extend the 32-bit result to 64 bits.  RV64 only.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # sraiw x1, x2, 4  — x1 = sext32(x2[31:0] >>a 4)
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    shamt = imm & 0x1F  # Lower 5 bits for 32-bit shift
    # Arithmetic right shift on 32-bit value
    if rs1_val & 0x80000000:
        # Negative number, preserve sign
        mask = (1 << (32 - shamt)) - 1
        mask = ~mask
        result_32 = ((rs1_val >> shamt) | mask) & 0xFFFFFFFF
    else:
        result_32 = (rs1_val >> shamt) & 0xFFFFFFFF
    # Sign extend from 32 bits
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32

    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes
