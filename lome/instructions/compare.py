# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Compare instruction implementations: SLT, SLTI, SLTU, SLTIU."""

from __future__ import annotations

from lome.changes import ChangeRecord, GPRWrite
from lome.state import State


def execute_slt(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SLT: rd = (rs1 < rs2) ? 1 : 0 (signed comparison)

    Set rd to 1 if the signed value in rs1 is less than the signed value
    in rs2, otherwise set rd to 0.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``rs2``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # slt x1, x2, x3  — x1 = 1 if x2 < x3 (signed), else 0
    """
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    # Signed comparison: treat as signed integers
    rs1_signed = (
        rs1_val if rs1_val < 0x8000000000000000 else rs1_val - 0x10000000000000000
    )
    rs2_signed = (
        rs2_val if rs2_val < 0x8000000000000000 else rs2_val - 0x10000000000000000
    )
    result = 1 if rs1_signed < rs2_signed else 0

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_slti(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SLTI: rd = (rs1 < imm) ? 1 : 0 (signed comparison)

    Set rd to 1 if the signed value in rs1 is less than the sign-extended
    immediate, otherwise set rd to 0.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # slti x1, x2, 10  — x1 = 1 if x2 < 10 (signed), else 0
    """
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    # Signed comparison: treat as signed integers
    rs1_signed = (
        rs1_val if rs1_val < 0x8000000000000000 else rs1_val - 0x10000000000000000
    )
    imm_signed = imm if imm < 0x8000000000000000 else imm - 0x10000000000000000
    result = 1 if rs1_signed < imm_signed else 0

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_sltu(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SLTU: rd = (rs1 < rs2) ? 1 : 0 (unsigned comparison)

    Set rd to 1 if the unsigned value in rs1 is less than the unsigned
    value in rs2, otherwise set rd to 0.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``rs2``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # sltu x1, x2, x3  — x1 = 1 if x2 < x3 (unsigned), else 0
    """
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    # Unsigned comparison
    result = 1 if rs1_val < rs2_val else 0

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_sltiu(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SLTIU: rd = (rs1 < imm) ? 1 : 0 (unsigned comparison)

    Set rd to 1 if the unsigned value in rs1 is less than the
    zero-extended immediate, otherwise set rd to 0.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # sltiu x1, x2, 5  — x1 = 1 if x2 < 5 (unsigned), else 0
    """
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    # Treat immediate as unsigned (zero-extend)
    imm_unsigned = imm & 0xFFFFFFFFFFFFFFFF
    # Unsigned comparison
    result = 1 if rs1_val < imm_unsigned else 0

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes
