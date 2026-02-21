# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Branch instruction implementations: BEQ, BNE, BLT, BGE, BLTU, BGEU."""

from __future__ import annotations

from lome.changes import BranchInfo, ChangeRecord
from lome.state import State


def execute_beq(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute BEQ: if (rs1 == rs2) pc += imm

    Branch to *pc + imm* when registers rs1 and rs2 are equal; otherwise
    fall through to *pc + 4*.

    Parameters:
        operand_values: dict with keys ``rs1``, ``rs2``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing branch info and pc change.

    Example::

        # beq x1, x2, 8  — if x1 == x2, jump forward 8 bytes
    """
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    taken = rs1_val == rs2_val
    target = pc + imm if taken else pc + 4

    changes = ChangeRecord()
    changes.branch_info = BranchInfo(taken=taken, target=target, condition="eq")
    changes.pc_change = (target, pc)
    return changes


def execute_bne(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute BNE: if (rs1 != rs2) pc += imm

    Branch to *pc + imm* when registers rs1 and rs2 are not equal;
    otherwise fall through to *pc + 4*.

    Parameters:
        operand_values: dict with keys ``rs1``, ``rs2``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing branch info and pc change.

    Example::

        # bne x1, x2, -4  — loop back 4 bytes while x1 != x2
    """
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    taken = rs1_val != rs2_val
    target = pc + imm if taken else pc + 4

    changes = ChangeRecord()
    changes.branch_info = BranchInfo(taken=taken, target=target, condition="ne")
    changes.pc_change = (target, pc)
    return changes


def execute_blt(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute BLT: if (rs1 < rs2) pc += imm (signed comparison)

    Branch to *pc + imm* when the signed value in rs1 is less than the
    signed value in rs2; otherwise fall through to *pc + 4*.

    Parameters:
        operand_values: dict with keys ``rs1``, ``rs2``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing branch info and pc change.

    Example::

        # blt x3, x4, 16  — branch forward 16 bytes if x3 < x4 (signed)
    """
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    # Signed comparison
    rs1_signed = (
        rs1_val if rs1_val < 0x8000000000000000 else rs1_val - 0x10000000000000000
    )
    rs2_signed = (
        rs2_val if rs2_val < 0x8000000000000000 else rs2_val - 0x10000000000000000
    )
    taken = rs1_signed < rs2_signed
    target = pc + imm if taken else pc + 4

    changes = ChangeRecord()
    changes.branch_info = BranchInfo(taken=taken, target=target, condition="lt")
    changes.pc_change = (target, pc)
    return changes


def execute_bge(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute BGE: if (rs1 >= rs2) pc += imm (signed comparison)

    Branch to *pc + imm* when the signed value in rs1 is greater than or
    equal to the signed value in rs2; otherwise fall through to *pc + 4*.

    Parameters:
        operand_values: dict with keys ``rs1``, ``rs2``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing branch info and pc change.

    Example::

        # bge x5, x6, 12  — branch forward 12 bytes if x5 >= x6 (signed)
    """
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    # Signed comparison
    rs1_signed = (
        rs1_val if rs1_val < 0x8000000000000000 else rs1_val - 0x10000000000000000
    )
    rs2_signed = (
        rs2_val if rs2_val < 0x8000000000000000 else rs2_val - 0x10000000000000000
    )
    taken = rs1_signed >= rs2_signed
    target = pc + imm if taken else pc + 4

    changes = ChangeRecord()
    changes.branch_info = BranchInfo(taken=taken, target=target, condition="ge")
    changes.pc_change = (target, pc)
    return changes


def execute_bltu(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute BLTU: if (rs1 < rs2) pc += imm (unsigned comparison)

    Branch to *pc + imm* when the unsigned value in rs1 is less than the
    unsigned value in rs2; otherwise fall through to *pc + 4*.

    Parameters:
        operand_values: dict with keys ``rs1``, ``rs2``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing branch info and pc change.

    Example::

        # bltu x7, x8, 20  — branch forward 20 bytes if x7 < x8 (unsigned)
    """
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    taken = rs1_val < rs2_val
    target = pc + imm if taken else pc + 4

    changes = ChangeRecord()
    changes.branch_info = BranchInfo(taken=taken, target=target, condition="ltu")
    changes.pc_change = (target, pc)
    return changes


def execute_bgeu(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute BGEU: if (rs1 >= rs2) pc += imm (unsigned comparison)

    Branch to *pc + imm* when the unsigned value in rs1 is greater than or
    equal to the unsigned value in rs2; otherwise fall through to *pc + 4*.

    Parameters:
        operand_values: dict with keys ``rs1``, ``rs2``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing branch info and pc change.

    Example::

        # bgeu x9, x10, -8  — loop back 8 bytes if x9 >= x10 (unsigned)
    """
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    taken = rs1_val >= rs2_val
    target = pc + imm if taken else pc + 4

    changes = ChangeRecord()
    changes.branch_info = BranchInfo(taken=taken, target=target, condition="geu")
    changes.pc_change = (target, pc)
    return changes
