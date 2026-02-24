# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Floating-point instruction implementations: F/D extension (flw, fsw, fadd, etc.)."""

from __future__ import annotations

import math
from typing import Optional

from lome.changes import ChangeRecord, FPRWrite, GPRRead, GPRWrite, MemoryAccess
from lome.float_utils import (
    bits_to_float_d,
    bits_to_float_s,
    effective_rounding_mode,
    float_to_bits_d,
    float_to_bits_s,
    round_for_float,
)
from lome.memory import MemoryInterface
from lome.state import State
from lome.types import OperandValues

_MASK_32 = 0xFFFF_FFFF
_MASK_64 = 0xFFFF_FFFF_FFFF_FFFF


# ---------------------------------------------------------------------------
# Load / Store
# ---------------------------------------------------------------------------


def execute_flw(
    operand_values: OperandValues,
    state: State,
    pc: int,
    *,
    memory: Optional[MemoryInterface] = None,
) -> ChangeRecord:
    """FLW: load single-precision from memory into FPR rd."""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm", 0)
    base = state.get_gpr(rs1_idx)
    addr = base + imm
    value = memory.load(addr, 4) if memory else 0
    value = value & _MASK_32
    changes = ChangeRecord()
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=base))
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=4, is_write=False)
    )
    old = state.set_fpr(rd, value)
    changes.fpr_writes.append(FPRWrite(register=rd, value=value, old_value=old))
    return changes


def execute_fsw(
    operand_values: OperandValues,
    state: State,
    pc: int,
    *,
    memory: Optional[MemoryInterface] = None,
) -> ChangeRecord:
    """FSW: store single-precision from FPR rs2 to memory."""
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    imm = operand_values.get("imm", 0)
    base = state.get_gpr(rs1_idx)
    addr = base + imm
    value = state.get_fpr(rs2_idx) & _MASK_32
    changes = ChangeRecord()
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=base))
    if memory:
        memory.store(addr, value, 4)
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=4, is_write=True)
    )
    return changes


def execute_fld(
    operand_values: OperandValues,
    state: State,
    pc: int,
    *,
    memory: Optional[MemoryInterface] = None,
) -> ChangeRecord:
    """FLD: load double-precision from memory into FPR rd."""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm", 0)
    base = state.get_gpr(rs1_idx)
    addr = base + imm
    value = memory.load(addr, 8) if memory else 0
    value = value & _MASK_64
    changes = ChangeRecord()
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=base))
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=8, is_write=False)
    )
    old = state.set_fpr(rd, value)
    changes.fpr_writes.append(FPRWrite(register=rd, value=value, old_value=old))
    return changes


def execute_fsd(
    operand_values: OperandValues,
    state: State,
    pc: int,
    *,
    memory: Optional[MemoryInterface] = None,
) -> ChangeRecord:
    """FSD: store double-precision from FPR rs2 to memory."""
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    imm = operand_values.get("imm", 0)
    base = state.get_gpr(rs1_idx)
    addr = base + imm
    value = state.get_fpr(rs2_idx) & _MASK_64
    changes = ChangeRecord()
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=base))
    if memory:
        memory.store(addr, value, 8)
    changes.memory_accesses.append(
        MemoryAccess(address=addr, value=value, size=8, is_write=True)
    )
    return changes


# ---------------------------------------------------------------------------
# Arithmetic (with rounding)
# ---------------------------------------------------------------------------


def _fadd_s(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    rs2 = operand_values.get("rs2")
    a = bits_to_float_s(state.get_fpr(rs1))
    b = bits_to_float_s(state.get_fpr(rs2))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(a + b, 24, rm)
    bits = float_to_bits_s(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def _fadd_d(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    rs2 = operand_values.get("rs2")
    a = bits_to_float_d(state.get_fpr(rs1))
    b = bits_to_float_d(state.get_fpr(rs2))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(a + b, 53, rm)
    bits = float_to_bits_d(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fadd_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fadd_s(operand_values, state)


def execute_fadd_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fadd_d(operand_values, state)


def _fsub_s(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    rs2 = operand_values.get("rs2")
    a = bits_to_float_s(state.get_fpr(rs1))
    b = bits_to_float_s(state.get_fpr(rs2))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(a - b, 24, rm)
    bits = float_to_bits_s(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def _fsub_d(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    rs2 = operand_values.get("rs2")
    a = bits_to_float_d(state.get_fpr(rs1))
    b = bits_to_float_d(state.get_fpr(rs2))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(a - b, 53, rm)
    bits = float_to_bits_d(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fsub_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fsub_s(operand_values, state)


def execute_fsub_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fsub_d(operand_values, state)


def _fmul_s(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2 = operand_values.get("rs1"), operand_values.get("rs2")
    a = bits_to_float_s(state.get_fpr(rs1))
    b = bits_to_float_s(state.get_fpr(rs2))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(a * b, 24, rm)
    bits = float_to_bits_s(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def _fmul_d(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2 = operand_values.get("rs1"), operand_values.get("rs2")
    a = bits_to_float_d(state.get_fpr(rs1))
    b = bits_to_float_d(state.get_fpr(rs2))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(a * b, 53, rm)
    bits = float_to_bits_d(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fmul_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fmul_s(operand_values, state)


def execute_fmul_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fmul_d(operand_values, state)


def _fdiv_s(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2 = operand_values.get("rs1"), operand_values.get("rs2")
    a = bits_to_float_s(state.get_fpr(rs1))
    b = bits_to_float_s(state.get_fpr(rs2))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(
        a / b if b != 0 else (float("inf") if a != 0 else float("nan")), 24, rm
    )
    bits = float_to_bits_s(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def _fdiv_d(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2 = operand_values.get("rs1"), operand_values.get("rs2")
    a = bits_to_float_d(state.get_fpr(rs1))
    b = bits_to_float_d(state.get_fpr(rs2))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(
        a / b if b != 0 else (float("inf") if a != 0 else float("nan")), 53, rm
    )
    bits = float_to_bits_d(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fdiv_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fdiv_s(operand_values, state)


def execute_fdiv_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fdiv_d(operand_values, state)


def _fsqrt_s(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    a = bits_to_float_s(state.get_fpr(rs1))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(math.sqrt(a) if a >= 0 else float("nan"), 24, rm)
    bits = float_to_bits_s(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def _fsqrt_d(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    a = bits_to_float_d(state.get_fpr(rs1))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(math.sqrt(a) if a >= 0 else float("nan"), 53, rm)
    bits = float_to_bits_d(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fsqrt_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fsqrt_s(operand_values, state)


def execute_fsqrt_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fsqrt_d(operand_values, state)


# ---------------------------------------------------------------------------
# Fused multiply-add: fmadd rd = rs1*rs2+rs3, fmsub rd = rs1*rs2-rs3,
# fnmadd rd = -(rs1*rs2+rs3), fnmsub rd = -(rs1*rs2-rs3)
# ---------------------------------------------------------------------------


def _fmadd_s(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2, rs3 = (
        operand_values.get("rs1"),
        operand_values.get("rs2"),
        operand_values.get("rs3"),
    )
    a = bits_to_float_s(state.get_fpr(rs1))
    b = bits_to_float_s(state.get_fpr(rs2))
    c = bits_to_float_s(state.get_fpr(rs3))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(a * b + c, 24, rm)
    bits = float_to_bits_s(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def _fmadd_d(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2, rs3 = (
        operand_values.get("rs1"),
        operand_values.get("rs2"),
        operand_values.get("rs3"),
    )
    a = bits_to_float_d(state.get_fpr(rs1))
    b = bits_to_float_d(state.get_fpr(rs2))
    c = bits_to_float_d(state.get_fpr(rs3))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(a * b + c, 53, rm)
    bits = float_to_bits_d(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fmadd_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fmadd_s(operand_values, state)


def execute_fmadd_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fmadd_d(operand_values, state)


def _fmsub_s(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2, rs3 = (
        operand_values.get("rs1"),
        operand_values.get("rs2"),
        operand_values.get("rs3"),
    )
    a = bits_to_float_s(state.get_fpr(rs1))
    b = bits_to_float_s(state.get_fpr(rs2))
    c = bits_to_float_s(state.get_fpr(rs3))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(a * b - c, 24, rm)
    bits = float_to_bits_s(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def _fmsub_d(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2, rs3 = (
        operand_values.get("rs1"),
        operand_values.get("rs2"),
        operand_values.get("rs3"),
    )
    a = bits_to_float_d(state.get_fpr(rs1))
    b = bits_to_float_d(state.get_fpr(rs2))
    c = bits_to_float_d(state.get_fpr(rs3))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(a * b - c, 53, rm)
    bits = float_to_bits_d(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fmsub_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fmsub_s(operand_values, state)


def execute_fmsub_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fmsub_d(operand_values, state)


def _fnmadd_s(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2, rs3 = (
        operand_values.get("rs1"),
        operand_values.get("rs2"),
        operand_values.get("rs3"),
    )
    a = bits_to_float_s(state.get_fpr(rs1))
    b = bits_to_float_s(state.get_fpr(rs2))
    c = bits_to_float_s(state.get_fpr(rs3))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(-(a * b + c), 24, rm)
    bits = float_to_bits_s(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def _fnmadd_d(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2, rs3 = (
        operand_values.get("rs1"),
        operand_values.get("rs2"),
        operand_values.get("rs3"),
    )
    a = bits_to_float_d(state.get_fpr(rs1))
    b = bits_to_float_d(state.get_fpr(rs2))
    c = bits_to_float_d(state.get_fpr(rs3))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(-(a * b + c), 53, rm)
    bits = float_to_bits_d(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fnmadd_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fnmadd_s(operand_values, state)


def execute_fnmadd_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fnmadd_d(operand_values, state)


def _fnmsub_s(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2, rs3 = (
        operand_values.get("rs1"),
        operand_values.get("rs2"),
        operand_values.get("rs3"),
    )
    a = bits_to_float_s(state.get_fpr(rs1))
    b = bits_to_float_s(state.get_fpr(rs2))
    c = bits_to_float_s(state.get_fpr(rs3))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(-(a * b - c), 24, rm)
    bits = float_to_bits_s(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def _fnmsub_d(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2, rs3 = (
        operand_values.get("rs1"),
        operand_values.get("rs2"),
        operand_values.get("rs3"),
    )
    a = bits_to_float_d(state.get_fpr(rs1))
    b = bits_to_float_d(state.get_fpr(rs2))
    c = bits_to_float_d(state.get_fpr(rs3))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(-(a * b - c), 53, rm)
    bits = float_to_bits_d(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fnmsub_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fnmsub_s(operand_values, state)


def execute_fnmsub_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fnmsub_d(operand_values, state)


# ---------------------------------------------------------------------------
# Sign injection: fsgnj (copy), fsgnjn (negate), fsgnjx (xor)
# ---------------------------------------------------------------------------


def _sign_bits_s(bits: int) -> int:
    return bits & (1 << 31)


def _sign_bits_d(bits: int) -> int:
    return bits & (1 << 63)


def _fsgnj_s(
    operand_values: OperandValues, state: State, neg: bool = False, xor: bool = False
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    rs2 = operand_values.get("rs2")
    v1 = state.get_fpr(rs1) & _MASK_32
    v2 = state.get_fpr(rs2) & _MASK_32
    s1 = _sign_bits_s(v1)
    s2 = _sign_bits_s(v2)
    if xor:
        new_s = (s1 ^ s2) & (1 << 31)
    else:
        new_s = (~s2 & (1 << 31)) if neg else s2
    result = (v1 & (_MASK_32 >> 1)) | new_s
    changes = ChangeRecord()
    old = state.set_fpr(rd, result)
    changes.fpr_writes.append(FPRWrite(register=rd, value=result, old_value=old))
    return changes


def _fsgnj_d(
    operand_values: OperandValues, state: State, neg: bool = False, xor: bool = False
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    rs2 = operand_values.get("rs2")
    v1 = state.get_fpr(rs1) & _MASK_64
    v2 = state.get_fpr(rs2) & _MASK_64
    s1 = _sign_bits_d(v1)
    s2 = _sign_bits_d(v2)
    if xor:
        new_s = (s1 ^ s2) & (1 << 63)
    else:
        new_s = (~s2 & (1 << 63)) if neg else s2
    result = (v1 & (_MASK_64 >> 1)) | new_s
    changes = ChangeRecord()
    old = state.set_fpr(rd, result)
    changes.fpr_writes.append(FPRWrite(register=rd, value=result, old_value=old))
    return changes


def execute_fsgnj_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fsgnj_s(operand_values, state, neg=False, xor=False)


def execute_fsgnj_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fsgnj_d(operand_values, state, neg=False, xor=False)


def execute_fsgnjn_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fsgnj_s(operand_values, state, neg=True, xor=False)


def execute_fsgnjn_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fsgnj_d(operand_values, state, neg=True, xor=False)


def execute_fsgnjx_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fsgnj_s(operand_values, state, neg=False, xor=True)


def execute_fsgnjx_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fsgnj_d(operand_values, state, neg=False, xor=True)


# ---------------------------------------------------------------------------
# Min / Max (no rounding; result is one of the inputs)
# ---------------------------------------------------------------------------


def _fmin_s(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd, rs1, rs2 = (
        operand_values.get("rd"),
        operand_values.get("rs1"),
        operand_values.get("rs2"),
    )
    a = bits_to_float_s(state.get_fpr(rs1))
    b = bits_to_float_s(state.get_fpr(rs2))
    if math.isnan(a) and math.isnan(b):
        res = float_to_bits_s(float("nan"))
    elif math.isnan(a):
        res = state.get_fpr(rs2) & _MASK_32
    elif math.isnan(b):
        res = state.get_fpr(rs1) & _MASK_32
    else:
        res = float_to_bits_s(min(a, b))
    changes = ChangeRecord()
    old = state.set_fpr(rd, res)
    changes.fpr_writes.append(FPRWrite(register=rd, value=res, old_value=old))
    return changes


def _fmin_d(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd, rs1, rs2 = (
        operand_values.get("rd"),
        operand_values.get("rs1"),
        operand_values.get("rs2"),
    )
    a = bits_to_float_d(state.get_fpr(rs1))
    b = bits_to_float_d(state.get_fpr(rs2))
    if math.isnan(a) and math.isnan(b):
        res = float_to_bits_d(float("nan"))
    elif math.isnan(a):
        res = state.get_fpr(rs2) & _MASK_64
    elif math.isnan(b):
        res = state.get_fpr(rs1) & _MASK_64
    else:
        res = float_to_bits_d(min(a, b))
    changes = ChangeRecord()
    old = state.set_fpr(rd, res)
    changes.fpr_writes.append(FPRWrite(register=rd, value=res, old_value=old))
    return changes


def execute_fmin_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fmin_s(operand_values, state)


def execute_fmin_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fmin_d(operand_values, state)


def _fmax_s(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd, rs1, rs2 = (
        operand_values.get("rd"),
        operand_values.get("rs1"),
        operand_values.get("rs2"),
    )
    a = bits_to_float_s(state.get_fpr(rs1))
    b = bits_to_float_s(state.get_fpr(rs2))
    if math.isnan(a) and math.isnan(b):
        res = float_to_bits_s(float("nan"))
    elif math.isnan(a):
        res = state.get_fpr(rs2) & _MASK_32
    elif math.isnan(b):
        res = state.get_fpr(rs1) & _MASK_32
    else:
        res = float_to_bits_s(max(a, b))
    changes = ChangeRecord()
    old = state.set_fpr(rd, res)
    changes.fpr_writes.append(FPRWrite(register=rd, value=res, old_value=old))
    return changes


def _fmax_d(operand_values: OperandValues, state: State) -> ChangeRecord:
    rd, rs1, rs2 = (
        operand_values.get("rd"),
        operand_values.get("rs1"),
        operand_values.get("rs2"),
    )
    a = bits_to_float_d(state.get_fpr(rs1))
    b = bits_to_float_d(state.get_fpr(rs2))
    if math.isnan(a) and math.isnan(b):
        res = float_to_bits_d(float("nan"))
    elif math.isnan(a):
        res = state.get_fpr(rs2) & _MASK_64
    elif math.isnan(b):
        res = state.get_fpr(rs1) & _MASK_64
    else:
        res = float_to_bits_d(max(a, b))
    changes = ChangeRecord()
    old = state.set_fpr(rd, res)
    changes.fpr_writes.append(FPRWrite(register=rd, value=res, old_value=old))
    return changes


def execute_fmax_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fmax_s(operand_values, state)


def execute_fmax_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    return _fmax_d(operand_values, state)


# ---------------------------------------------------------------------------
# Compare: feq, fle, flt -> write 0 or 1 to GPR rd
# ---------------------------------------------------------------------------


def execute_feq_s(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2 = operand_values.get("rs1"), operand_values.get("rs2")
    a = bits_to_float_s(state.get_fpr(rs1))
    b = bits_to_float_s(state.get_fpr(rs2))
    val = 1 if a == b else 0
    changes = ChangeRecord()
    old = state.set_gpr(rd, val)
    changes.gpr_writes.append(GPRWrite(register=rd, value=val, old_value=old))
    return changes


def execute_feq_d(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2 = operand_values.get("rs1"), operand_values.get("rs2")
    a = bits_to_float_d(state.get_fpr(rs1))
    b = bits_to_float_d(state.get_fpr(rs2))
    val = 1 if a == b else 0
    changes = ChangeRecord()
    old = state.set_gpr(rd, val)
    changes.gpr_writes.append(GPRWrite(register=rd, value=val, old_value=old))
    return changes


def execute_fle_s(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2 = operand_values.get("rs1"), operand_values.get("rs2")
    a = bits_to_float_s(state.get_fpr(rs1))
    b = bits_to_float_s(state.get_fpr(rs2))
    val = 1 if (not (math.isnan(a) or math.isnan(b)) and a <= b) else 0
    changes = ChangeRecord()
    old = state.set_gpr(rd, val)
    changes.gpr_writes.append(GPRWrite(register=rd, value=val, old_value=old))
    return changes


def execute_fle_d(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2 = operand_values.get("rs1"), operand_values.get("rs2")
    a = bits_to_float_d(state.get_fpr(rs1))
    b = bits_to_float_d(state.get_fpr(rs2))
    val = 1 if (not (math.isnan(a) or math.isnan(b)) and a <= b) else 0
    changes = ChangeRecord()
    old = state.set_gpr(rd, val)
    changes.gpr_writes.append(GPRWrite(register=rd, value=val, old_value=old))
    return changes


def execute_flt_s(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2 = operand_values.get("rs1"), operand_values.get("rs2")
    a = bits_to_float_s(state.get_fpr(rs1))
    b = bits_to_float_s(state.get_fpr(rs2))
    val = 1 if (not (math.isnan(a) or math.isnan(b)) and a < b) else 0
    changes = ChangeRecord()
    old = state.set_gpr(rd, val)
    changes.gpr_writes.append(GPRWrite(register=rd, value=val, old_value=old))
    return changes


def execute_flt_d(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1, rs2 = operand_values.get("rs1"), operand_values.get("rs2")
    a = bits_to_float_d(state.get_fpr(rs1))
    b = bits_to_float_d(state.get_fpr(rs2))
    val = 1 if (not (math.isnan(a) or math.isnan(b)) and a < b) else 0
    changes = ChangeRecord()
    old = state.set_gpr(rd, val)
    changes.gpr_writes.append(GPRWrite(register=rd, value=val, old_value=old))
    return changes


# ---------------------------------------------------------------------------
# Move: fmv.w.x (GPR->FPR), fmv.d.x (GPR->FPR), fmv.x.w (FPR->GPR), fmv.x.d (FPR->GPR)
# ---------------------------------------------------------------------------


def execute_fmv_w_x(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    raw = state.get_gpr(rs1)
    bits = raw & _MASK_32
    changes = ChangeRecord()
    if rs1 is not None:
        changes.gpr_reads.append(GPRRead(register=rs1, value=raw))
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fmv_d_x(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    raw = state.get_gpr(rs1)
    bits = raw & _MASK_64
    changes = ChangeRecord()
    if rs1 is not None:
        changes.gpr_reads.append(GPRRead(register=rs1, value=raw))
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fmv_x_w(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    bits = state.get_fpr(rs1) & _MASK_32
    # Sign-extend to 64 bits
    if bits & (1 << 31):
        bits = bits | 0xFFFF_FFFF_0000_0000
    changes = ChangeRecord()
    old = state.set_gpr(rd, bits)
    changes.gpr_writes.append(GPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fmv_x_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    bits = state.get_fpr(rs1) & _MASK_64
    changes = ChangeRecord()
    old = state.set_gpr(rd, bits)
    changes.gpr_writes.append(GPRWrite(register=rd, value=bits, old_value=old))
    return changes


# ---------------------------------------------------------------------------
# FCVT: float<->int and float<->float (with rounding where needed)
# ---------------------------------------------------------------------------


def execute_fcvt_s_w(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    x = state.get_gpr(rs1)
    changes = ChangeRecord()
    if rs1 is not None:
        changes.gpr_reads.append(GPRRead(register=rs1, value=x))
    if x >= (1 << 31):
        x = x - (1 << 32)
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(float(x), 24, rm)
    bits = float_to_bits_s(res)
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fcvt_s_wu(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    raw = state.get_gpr(rs1)
    x = raw & _MASK_32
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(float(x), 24, rm)
    bits = float_to_bits_s(res)
    changes = ChangeRecord()
    if rs1 is not None:
        changes.gpr_reads.append(GPRRead(register=rs1, value=raw))
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fcvt_s_l(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    raw = state.get_gpr(rs1)
    x = raw & _MASK_64
    if x >= (1 << 63):
        x = x - (1 << 64)
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(float(x), 24, rm)
    bits = float_to_bits_s(res)
    changes = ChangeRecord()
    if rs1 is not None:
        changes.gpr_reads.append(GPRRead(register=rs1, value=raw))
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fcvt_s_lu(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    raw = state.get_gpr(rs1)
    x = raw & _MASK_64
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(float(x), 24, rm)
    bits = float_to_bits_s(res)
    changes = ChangeRecord()
    if rs1 is not None:
        changes.gpr_reads.append(GPRRead(register=rs1, value=raw))
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fcvt_d_w(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    x = state.get_gpr(rs1)
    changes = ChangeRecord()
    if rs1 is not None:
        changes.gpr_reads.append(GPRRead(register=rs1, value=x))
    if x >= (1 << 31):
        x = x - (1 << 32)
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(float(x), 53, rm)
    bits = float_to_bits_d(res)
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fcvt_d_wu(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    raw = state.get_gpr(rs1)
    x = raw & _MASK_32
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(float(x), 53, rm)
    bits = float_to_bits_d(res)
    changes = ChangeRecord()
    if rs1 is not None:
        changes.gpr_reads.append(GPRRead(register=rs1, value=raw))
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fcvt_d_l(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    raw = state.get_gpr(rs1)
    x = raw & _MASK_64
    if x >= (1 << 63):
        x = x - (1 << 64)
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(float(x), 53, rm)
    bits = float_to_bits_d(res)
    changes = ChangeRecord()
    if rs1 is not None:
        changes.gpr_reads.append(GPRRead(register=rs1, value=raw))
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fcvt_d_lu(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    raw = state.get_gpr(rs1)
    x = raw & _MASK_64
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(float(x), 53, rm)
    bits = float_to_bits_d(res)
    changes = ChangeRecord()
    if rs1 is not None:
        changes.gpr_reads.append(GPRRead(register=rs1, value=raw))
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fcvt_w_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    f = bits_to_float_s(state.get_fpr(rs1))
    rm = effective_rounding_mode(operand_values, state)
    if math.isnan(f):
        x = 0x8000_0000
    elif math.isinf(f):
        x = 0x7FFF_FFFF if f > 0 else 0x8000_0000
    else:
        x = int(round_for_float(f, 24, rm))
        x = max(-(1 << 31), min((1 << 31) - 1, x))
    x = x & _MASK_64
    if x >= (1 << 31):
        x = x - (1 << 32)
    changes = ChangeRecord()
    old = state.set_gpr(rd, x)
    changes.gpr_writes.append(GPRWrite(register=rd, value=x, old_value=old))
    return changes


def execute_fcvt_wu_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    f = bits_to_float_s(state.get_fpr(rs1))
    rm = effective_rounding_mode(operand_values, state)
    if math.isnan(f) or f < 0:
        x = 0
    elif math.isinf(f):
        x = 0xFFFF_FFFF
    else:
        x = int(round_for_float(f, 24, rm))
        x = max(0, min(0xFFFF_FFFF, x))
        x = x & _MASK_64
    changes = ChangeRecord()
    old = state.set_gpr(rd, x)
    changes.gpr_writes.append(GPRWrite(register=rd, value=x, old_value=old))
    return changes


def execute_fcvt_w_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    f = bits_to_float_d(state.get_fpr(rs1))
    rm = effective_rounding_mode(operand_values, state)
    if math.isnan(f):
        x = 0x8000_0000
    elif math.isinf(f):
        x = 0x7FFF_FFFF if f > 0 else 0x8000_0000
    else:
        x = int(round_for_float(f, 53, rm))
        x = max(-(1 << 31), min((1 << 31) - 1, x))
    x = x & _MASK_64
    if x >= (1 << 31):
        x = x - (1 << 32)
    changes = ChangeRecord()
    old = state.set_gpr(rd, x)
    changes.gpr_writes.append(GPRWrite(register=rd, value=x, old_value=old))
    return changes


def execute_fcvt_wu_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    f = bits_to_float_d(state.get_fpr(rs1))
    rm = effective_rounding_mode(operand_values, state)
    if math.isnan(f) or f < 0:
        x = 0
    elif math.isinf(f):
        x = 0xFFFF_FFFF
    else:
        x = int(round_for_float(f, 53, rm))
        x = max(0, min(0xFFFF_FFFF, x))
        x = x & _MASK_64
    changes = ChangeRecord()
    old = state.set_gpr(rd, x)
    changes.gpr_writes.append(GPRWrite(register=rd, value=x, old_value=old))
    return changes


def execute_fcvt_l_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    f = bits_to_float_s(state.get_fpr(rs1))
    rm = effective_rounding_mode(operand_values, state)
    if math.isnan(f):
        x = 0x8000_0000_0000_0000
    elif math.isinf(f):
        x = 0x7FFF_FFFF_FFFF_FFFF if f > 0 else 0x8000_0000_0000_0000
    else:
        x = int(round_for_float(f, 24, rm))
        x = max(-(1 << 63), min((1 << 63) - 1, x))
    x = x & _MASK_64
    changes = ChangeRecord()
    old = state.set_gpr(rd, x)
    changes.gpr_writes.append(GPRWrite(register=rd, value=x, old_value=old))
    return changes


def execute_fcvt_lu_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    f = bits_to_float_s(state.get_fpr(rs1))
    rm = effective_rounding_mode(operand_values, state)
    if math.isnan(f) or f < 0:
        x = 0
    elif math.isinf(f):
        x = 0xFFFF_FFFF_FFFF_FFFF
    else:
        x = int(round_for_float(f, 24, rm))
        x = max(0, min(0xFFFF_FFFF_FFFF_FFFF, x)) & _MASK_64
    changes = ChangeRecord()
    old = state.set_gpr(rd, x)
    changes.gpr_writes.append(GPRWrite(register=rd, value=x, old_value=old))
    return changes


def execute_fcvt_l_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    f = bits_to_float_d(state.get_fpr(rs1))
    rm = effective_rounding_mode(operand_values, state)
    if math.isnan(f):
        x = 0x8000_0000_0000_0000
    elif math.isinf(f):
        x = 0x7FFF_FFFF_FFFF_FFFF if f > 0 else 0x8000_0000_0000_0000
    else:
        x = int(round_for_float(f, 53, rm))
        x = max(-(1 << 63), min((1 << 63) - 1, x))
    x = x & _MASK_64
    changes = ChangeRecord()
    old = state.set_gpr(rd, x)
    changes.gpr_writes.append(GPRWrite(register=rd, value=x, old_value=old))
    return changes


def execute_fcvt_lu_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    f = bits_to_float_d(state.get_fpr(rs1))
    rm = effective_rounding_mode(operand_values, state)
    if math.isnan(f) or f < 0:
        x = 0
    elif math.isinf(f):
        x = 0xFFFF_FFFF_FFFF_FFFF
    else:
        x = int(round_for_float(f, 53, rm))
        x = max(0, min(0xFFFF_FFFF_FFFF_FFFF, x)) & _MASK_64
    changes = ChangeRecord()
    old = state.set_gpr(rd, x)
    changes.gpr_writes.append(GPRWrite(register=rd, value=x, old_value=old))
    return changes


def execute_fcvt_s_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    f = bits_to_float_d(state.get_fpr(rs1))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(f, 24, rm)
    bits = float_to_bits_s(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


def execute_fcvt_d_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    f = bits_to_float_s(state.get_fpr(rs1))
    rm = effective_rounding_mode(operand_values, state)
    res = round_for_float(f, 53, rm)
    bits = float_to_bits_d(res)
    changes = ChangeRecord()
    old = state.set_fpr(rd, bits)
    changes.fpr_writes.append(FPRWrite(register=rd, value=bits, old_value=old))
    return changes


# ---------------------------------------------------------------------------
# FCLASS: classification mask -> GPR rd
# ---------------------------------------------------------------------------


# Smallest positive normal: single 2**-126, double 2**-1022 (for subnormal detection)
_MIN_NORMAL_S = 2.0**-126
_MIN_NORMAL_D = 2.0**-1022


def _fclass_s(bits: int) -> int:
    f = bits_to_float_s(bits)
    mask = 0
    if math.isnan(f):
        if bits & (1 << 22):  # quiet NaN
            mask |= 1 << 9
        else:
            mask |= 1 << 8
    elif math.isinf(f):
        mask |= 1 << 0 if f < 0 else 1 << 7  # negative / positive infinity
    elif f == 0:
        mask |= 1 << 3 if (bits & (1 << 31)) else 1 << 4  # neg zero / pos zero
    else:
        abs_f = abs(f)
        if abs_f < _MIN_NORMAL_S:
            mask |= 1 << 2 if f < 0 else 1 << 5  # negative / positive subnormal
        else:
            mask |= 1 << 1 if f < 0 else 1 << 6  # negative / positive normal
    return mask


def _fclass_d(bits: int) -> int:
    f = bits_to_float_d(bits)
    mask = 0
    if math.isnan(f):
        if bits & (1 << 51):
            mask |= 1 << 9
        else:
            mask |= 1 << 8
    elif math.isinf(f):
        mask |= 1 << 0 if f < 0 else 1 << 7  # negative / positive infinity
    elif f == 0:
        mask |= 1 << 3 if (bits & (1 << 63)) else 1 << 4  # neg zero / pos zero
    else:
        abs_f = abs(f)
        if abs_f < _MIN_NORMAL_D:
            mask |= 1 << 2 if f < 0 else 1 << 5  # negative / positive subnormal
        else:
            mask |= 1 << 1 if f < 0 else 1 << 6  # negative / positive normal
    return mask


def execute_fclass_s(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    bits = state.get_fpr(rs1) & _MASK_32
    mask = _fclass_s(bits)
    changes = ChangeRecord()
    old = state.set_gpr(rd, mask)
    changes.gpr_writes.append(GPRWrite(register=rd, value=mask, old_value=old))
    return changes


def execute_fclass_d(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    rd = operand_values.get("rd")
    rs1 = operand_values.get("rs1")
    bits = state.get_fpr(rs1) & _MASK_64
    mask = _fclass_d(bits)
    changes = ChangeRecord()
    old = state.set_gpr(rd, mask)
    changes.gpr_writes.append(GPRWrite(register=rd, value=mask, old_value=old))
    return changes
