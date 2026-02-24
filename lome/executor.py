# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Instruction executor: routes decoded instructions to their handler functions.

The :func:`execute_instruction` function is the central dispatch point.  Given
a decoded :class:`InstructionInstance` (from Eumos), it looks up the mnemonic
in a handler table and delegates execution.  Unknown mnemonics produce a
``ChangeRecord`` with an ``illegal_instruction`` exception.

Speculation is handled transparently: the executor snapshots state before
execution and restores it afterwards so the caller sees no mutations.

Example -- direct use (normally called by :class:`Lome`)::

    from lome.executor import execute_instruction
    changes = execute_instruction(instance, state, pc, speculate=False)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from lome.changes import ChangeRecord
from lome.instructions import (
    arithmetic,
    branch,
    compare,
    jump,
    load_store,
    logical,
    shift,
    system,
)
from lome.instructions import (
    float as float_ins,
)
from lome.memory import MemoryInterface
from lome.ras import RASModel
from lome.state import State
from lome.types import OperandValues

_LOAD_STORE_MNEMONICS: frozenset[str] = frozenset(
    {
        "lb",
        "lh",
        "lw",
        "lbu",
        "lhu",
        "ld",
        "lwu",
        "sb",
        "sh",
        "sw",
        "sd",
        "flw",
        "fsw",
        "fld",
        "fsd",
    }
)
_JUMP_MNEMONICS: frozenset[str] = frozenset({"jal", "jalr"})

# ---------------------------------------------------------------------------
# Handler table
# ---------------------------------------------------------------------------

InstructionHandler = Callable[..., ChangeRecord]


_INSTRUCTION_HANDLERS: dict[str, InstructionHandler] = {
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
    # Float (F/D extension)
    "flw": float_ins.execute_flw,
    "fsw": float_ins.execute_fsw,
    "fld": float_ins.execute_fld,
    "fsd": float_ins.execute_fsd,
    "fadd.s": float_ins.execute_fadd_s,
    "fadd.d": float_ins.execute_fadd_d,
    "fsub.s": float_ins.execute_fsub_s,
    "fsub.d": float_ins.execute_fsub_d,
    "fmul.s": float_ins.execute_fmul_s,
    "fmul.d": float_ins.execute_fmul_d,
    "fdiv.s": float_ins.execute_fdiv_s,
    "fdiv.d": float_ins.execute_fdiv_d,
    "fsqrt.s": float_ins.execute_fsqrt_s,
    "fsqrt.d": float_ins.execute_fsqrt_d,
    "fmadd.s": float_ins.execute_fmadd_s,
    "fmadd.d": float_ins.execute_fmadd_d,
    "fmsub.s": float_ins.execute_fmsub_s,
    "fmsub.d": float_ins.execute_fmsub_d,
    "fnmadd.s": float_ins.execute_fnmadd_s,
    "fnmadd.d": float_ins.execute_fnmadd_d,
    "fnmsub.s": float_ins.execute_fnmsub_s,
    "fnmsub.d": float_ins.execute_fnmsub_d,
    "fsgnj.s": float_ins.execute_fsgnj_s,
    "fsgnj.d": float_ins.execute_fsgnj_d,
    "fsgnjn.s": float_ins.execute_fsgnjn_s,
    "fsgnjn.d": float_ins.execute_fsgnjn_d,
    "fsgnjx.s": float_ins.execute_fsgnjx_s,
    "fsgnjx.d": float_ins.execute_fsgnjx_d,
    "fmin.s": float_ins.execute_fmin_s,
    "fmin.d": float_ins.execute_fmin_d,
    "fmax.s": float_ins.execute_fmax_s,
    "fmax.d": float_ins.execute_fmax_d,
    "feq.s": float_ins.execute_feq_s,
    "feq.d": float_ins.execute_feq_d,
    "fle.s": float_ins.execute_fle_s,
    "fle.d": float_ins.execute_fle_d,
    "flt.s": float_ins.execute_flt_s,
    "flt.d": float_ins.execute_flt_d,
    "fmv.w.x": float_ins.execute_fmv_w_x,
    "fmv.d.x": float_ins.execute_fmv_d_x,
    "fmv.x.w": float_ins.execute_fmv_x_w,
    "fmv.x.d": float_ins.execute_fmv_x_d,
    "fcvt.s.w": float_ins.execute_fcvt_s_w,
    "fcvt.s.wu": float_ins.execute_fcvt_s_wu,
    "fcvt.s.l": float_ins.execute_fcvt_s_l,
    "fcvt.s.lu": float_ins.execute_fcvt_s_lu,
    "fcvt.d.w": float_ins.execute_fcvt_d_w,
    "fcvt.d.wu": float_ins.execute_fcvt_d_wu,
    "fcvt.d.l": float_ins.execute_fcvt_d_l,
    "fcvt.d.lu": float_ins.execute_fcvt_d_lu,
    "fcvt.w.s": float_ins.execute_fcvt_w_s,
    "fcvt.wu.s": float_ins.execute_fcvt_wu_s,
    "fcvt.w.d": float_ins.execute_fcvt_w_d,
    "fcvt.wu.d": float_ins.execute_fcvt_wu_d,
    "fcvt.l.s": float_ins.execute_fcvt_l_s,
    "fcvt.lu.s": float_ins.execute_fcvt_lu_s,
    "fcvt.l.d": float_ins.execute_fcvt_l_d,
    "fcvt.lu.d": float_ins.execute_fcvt_lu_d,
    "fcvt.s.d": float_ins.execute_fcvt_s_d,
    "fcvt.d.s": float_ins.execute_fcvt_d_s,
    "fclass.s": float_ins.execute_fclass_s,
    "fclass.d": float_ins.execute_fclass_d,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def execute_instruction(
    instruction_instance: Any,
    state: State,
    pc: int,
    *,
    memory: Optional[MemoryInterface] = None,
    ras: Optional[RASModel] = None,
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
    memory : MemoryInterface or None, optional
        If provided, load/store instructions use it for reads/writes.
        Otherwise loads return 0 (placeholder).
    ras : RASModel or None, optional
        If provided, JAL/JALR update the return address stack.
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
    :meth:`Lome.execute`, but can be used directly for low-level
    testing::

        from eumos import Eumos
        from lome.executor import execute_instruction
        from lome.state import State

        state = State(Eumos())
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
        changes.exception_code = 2  # RISC-V mcause: Illegal instruction
        changes.exception = f"illegal_instruction: {mnemonic}"
        return changes

    kwargs: dict[str, Any] = {}
    if memory is not None and mnemonic in _LOAD_STORE_MNEMONICS:
        kwargs["memory"] = memory
    if ras is not None and mnemonic in _JUMP_MNEMONICS:
        kwargs["ras"] = ras

    if speculate:
        # Snapshot → execute → restore
        snapshot = state.snapshot_for_speculation()
        try:
            return handler(instruction_instance.operand_values, state, pc, **kwargs)
        finally:
            state.restore_from_speculation(snapshot)

    # Normal execution -- state is mutated in-place.
    # Note: PC advancement is handled by the model, not here.
    operand_values: OperandValues = instruction_instance.operand_values
    return handler(operand_values, state, pc, **kwargs)
