# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred. All Rights Reserved

"""Branch instruction implementations: BEQ, BNE, BLT, BGE, BLTU, BGEU."""

from riscv_model.changes import BranchInfo, ChangeRecord
from riscv_model.state import State


def execute_beq(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute BEQ: if (rs1 == rs2) pc += imm"""
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
    """Execute BNE: if (rs1 != rs2) pc += imm"""
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
    """Execute BLT: if (rs1 < rs2) pc += imm (signed comparison)"""
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    # Signed comparison
    rs1_signed = rs1_val if rs1_val < 0x8000000000000000 else rs1_val - 0x10000000000000000
    rs2_signed = rs2_val if rs2_val < 0x8000000000000000 else rs2_val - 0x10000000000000000
    taken = rs1_signed < rs2_signed
    target = pc + imm if taken else pc + 4

    changes = ChangeRecord()
    changes.branch_info = BranchInfo(taken=taken, target=target, condition="lt")
    changes.pc_change = (target, pc)
    return changes


def execute_bge(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute BGE: if (rs1 >= rs2) pc += imm (signed comparison)"""
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    # Signed comparison
    rs1_signed = rs1_val if rs1_val < 0x8000000000000000 else rs1_val - 0x10000000000000000
    rs2_signed = rs2_val if rs2_val < 0x8000000000000000 else rs2_val - 0x10000000000000000
    taken = rs1_signed >= rs2_signed
    target = pc + imm if taken else pc + 4

    changes = ChangeRecord()
    changes.branch_info = BranchInfo(taken=taken, target=target, condition="ge")
    changes.pc_change = (target, pc)
    return changes


def execute_bltu(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute BLTU: if (rs1 < rs2) pc += imm (unsigned comparison)"""
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
    """Execute BGEU: if (rs1 >= rs2) pc += imm (unsigned comparison)"""
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
