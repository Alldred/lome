# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Compare instruction implementations: SLT, SLTI, SLTU, SLTIU."""

from __future__ import annotations

from lome.changes import ChangeRecord
from lome.instructions.common import read_gpr, signed64, write_gpr
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
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = read_gpr(changes, state, rs1_idx)
    rs2_val = read_gpr(changes, state, rs2_idx)
    rs1_signed = signed64(rs1_val)
    rs2_signed = signed64(rs2_val)
    result = 1 if rs1_signed < rs2_signed else 0
    write_gpr(changes, state, rd, result)
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
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = read_gpr(changes, state, rs1_idx)
    rs1_signed = signed64(rs1_val)
    imm_signed = signed64(imm)
    result = 1 if rs1_signed < imm_signed else 0
    write_gpr(changes, state, rd, result)
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
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = read_gpr(changes, state, rs1_idx)
    rs2_val = read_gpr(changes, state, rs2_idx)
    result = 1 if rs1_val < rs2_val else 0
    write_gpr(changes, state, rd, result)
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
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = read_gpr(changes, state, rs1_idx)
    imm_unsigned = imm & 0xFFFFFFFFFFFFFFFF
    result = 1 if rs1_val < imm_unsigned else 0
    write_gpr(changes, state, rd, result)
    return changes
