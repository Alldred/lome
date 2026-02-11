# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Instruction executor: routes decoded instructions to their handler functions.

The :func:`execute_instruction` function is the central dispatch point.  Given
a decoded :class:`InstructionInstance` (from Eumos), it looks up the mnemonic
in a handler table and delegates execution.  Unknown mnemonics produce a
``ChangeRecord`` with an ``illegal_instruction`` exception.

Speculation is handled transparently: the executor snapshots state before
execution and restores it afterwards so the caller sees no mutations.

Example -- direct use (normally called by :class:`RISCVModel`)::

    from riscv_model.executor import execute_instruction
    changes = execute_instruction(instance, state, pc, speculate=False)
"""

from __future__ import annotations

from typing import Any, Optional

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

# ---------------------------------------------------------------------------
# Handler table
# ---------------------------------------------------------------------------

_INSTRUCTION_HANDLERS: dict[str, Any] = {
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def execute_instruction(
    instruction_instance: Any,
    state: State,
    pc: int,
    *,
    speculate: bool = False,
) -> Optional[ChangeRecord]:
    """Execute a decoded instruction instance.

    Parameters
    ----------
    instruction_instance
        An :class:`InstructionInstance` from the Eumos decoder, or ``None``
        if decoding failed.
    state : State
        The architectural state to read from and (unless speculating) write to.
    pc : int
        Current program counter value.
    speculate : bool, optional
        If ``True``, snapshot state before execution and restore it
        afterwards so the caller sees no mutations (default ``False``).

    Returns
    -------
    ChangeRecord or None
        A record of all changes the instruction made (or would make).
        Returns ``None`` only if *instruction_instance* is ``None``.

    Examples
    --------
    This function is normally called internally by
    :meth:`RISCVModel.execute`, but can be used directly for low-level
    testing::

        from eumos import load_all_gprs, load_all_csrs
        from riscv_model.executor import execute_instruction
        from riscv_model.state import State

        gprs = load_all_gprs()
        csrs = load_all_csrs()
        state = State(gpr_defs=gprs, csr_defs=csrs)
        # (decode an instruction with Eumos to get instance)
        changes = execute_instruction(instance, state, pc=0)
    """
    if instruction_instance is None:
        return None

    mnemonic: str = instruction_instance.instruction.mnemonic
    handler = _INSTRUCTION_HANDLERS.get(mnemonic)

    if handler is None:
        # Unknown / unsupported instruction
        changes = ChangeRecord()
        changes.exception = f"illegal_instruction: {mnemonic}"
        return changes

    if speculate:
        # Snapshot → execute → restore
        snapshot = state.snapshot()
        try:
            changes = handler(instruction_instance.operand_values, state, pc)
            return changes
        except Exception as exc:
            changes = ChangeRecord()
            changes.exception = f"execution_error: {exc}"
            return changes
        finally:
            state.restore(snapshot)
    else:
        # Normal execution -- state is mutated in-place
        # Note: PC advancement is handled by the model, not here.
        try:
            return handler(instruction_instance.operand_values, state, pc)
        except Exception as exc:
            changes = ChangeRecord()
            changes.exception = f"execution_error: {exc}"
            return changes
