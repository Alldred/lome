# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred. All Rights Reserved

"""Instruction executor: routes decoded instructions to appropriate handlers."""

from typing import Optional

from riscv_model.changes import ChangeRecord
from riscv_model.instructions import (
    arithmetic,
    branch,
    compare,
    jump,
    load_store,
    logical,
    shift,
    system,
)
from riscv_model.state import State

# Instruction handler mapping: mnemonic -> function
_INSTRUCTION_HANDLERS = {
    # Arithmetic
    "add": arithmetic.execute_add,
    "addi": arithmetic.execute_addi,
    "sub": arithmetic.execute_sub,
    "addw": arithmetic.execute_addw,
    "addiw": arithmetic.execute_addiw,
    "subw": arithmetic.execute_subw,
    # Logical
    "and": logical.execute_and,
    "andi": logical.execute_andi,
    "or": logical.execute_or,
    "ori": logical.execute_ori,
    "xor": logical.execute_xor,
    "xori": logical.execute_xori,
    # Shift
    "sll": shift.execute_sll,
    "slli": shift.execute_slli,
    "srl": shift.execute_srl,
    "srli": shift.execute_srli,
    "sra": shift.execute_sra,
    "srai": shift.execute_srai,
    "sllw": shift.execute_sllw,
    "slliw": shift.execute_slliw,
    "srlw": shift.execute_srlw,
    "srliw": shift.execute_srliw,
    "sraw": shift.execute_sraw,
    "sraiw": shift.execute_sraiw,
    # Compare
    "slt": compare.execute_slt,
    "slti": compare.execute_slti,
    "sltu": compare.execute_sltu,
    "sltiu": compare.execute_sltiu,
    # Branch
    "beq": branch.execute_beq,
    "bne": branch.execute_bne,
    "blt": branch.execute_blt,
    "bge": branch.execute_bge,
    "bltu": branch.execute_bltu,
    "bgeu": branch.execute_bgeu,
    # Jump
    "jal": jump.execute_jal,
    "jalr": jump.execute_jalr,
    # Load/Store
    "lb": load_store.execute_lb,
    "lh": load_store.execute_lh,
    "lw": load_store.execute_lw,
    "lbu": load_store.execute_lbu,
    "lhu": load_store.execute_lhu,
    "ld": load_store.execute_ld,
    "lwu": load_store.execute_lwu,
    "sb": load_store.execute_sb,
    "sh": load_store.execute_sh,
    "sw": load_store.execute_sw,
    "sd": load_store.execute_sd,
    # System
    "lui": system.execute_lui,
    "auipc": system.execute_auipc,
    "csrrw": system.execute_csrrw,
    "csrrs": system.execute_csrrs,
    "csrrc": system.execute_csrrc,
    "csrrwi": system.execute_csrrwi,
    "csrrsi": system.execute_csrrsi,
    "csrrci": system.execute_csrrci,
    "ecall": system.execute_ecall,
    "ebreak": system.execute_ebreak,
    "mret": system.execute_mret,
    "fence": system.execute_fence,
    "fence.tso": system.execute_fence_tso,
}


def execute_instruction(
    instruction_instance, state: State, pc: int, speculate: bool = False
) -> Optional[ChangeRecord]:
    """Execute an instruction instance.

    Args:
        instruction_instance: InstructionInstance from slate decoder
        state: State object to read/write
        pc: Current program counter
        speculate: If True, don't modify state, just return changes

    Returns:
        ChangeRecord with all changes, or None if instruction unknown
    """
    if instruction_instance is None:
        return None

    mnemonic = instruction_instance.instruction.mnemonic
    handler = _INSTRUCTION_HANDLERS.get(mnemonic)

    if handler is None:
        # Unknown instruction
        changes = ChangeRecord()
        changes.exception = f"illegal_instruction: {mnemonic}"
        return changes

    if speculate:
        # Create snapshot, execute, restore
        snapshot = state.snapshot()
        try:
            changes = handler(instruction_instance.operand_values, state, pc)
            state.restore(snapshot)
            return changes
        except Exception as e:
            state.restore(snapshot)
            changes = ChangeRecord()
            changes.exception = f"execution_error: {str(e)}"
            return changes
    else:
        # Normal execution - modify state
        # Note: PC update is handled by the model, not here
        try:
            changes = handler(instruction_instance.operand_values, state, pc)
            return changes
        except Exception as e:
            changes = ChangeRecord()
            changes.exception = f"execution_error: {str(e)}"
            return changes
