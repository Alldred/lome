# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""M-extension instruction implementations.

Supported instructions:
MUL, MULH, MULHSU, MULHU, MULW,
DIV, DIVU, REM, REMU, DIVW, DIVUW, REMW, REMUW.
"""

from __future__ import annotations

from lome.changes import ChangeRecord
from lome.instructions.common import read_gpr, sext32, write_gpr
from lome.state import State
from lome.types import OperandValues

_MASK_32 = 0xFFFF_FFFF
_MASK_64 = 0xFFFF_FFFF_FFFF_FFFF
_SIGN_32 = 0x8000_0000
_SIGN_64 = 0x8000_0000_0000_0000
_INT_32_MIN = -(1 << 31)
_INT_64_MIN = -(1 << 63)


def _signed_32(value: int) -> int:
    value_32 = value & _MASK_32
    if value_32 & _SIGN_32:
        return value_32 - (1 << 32)
    return value_32


def _signed_64(value: int) -> int:
    value_64 = value & _MASK_64
    if value_64 & _SIGN_64:
        return value_64 - (1 << 64)
    return value_64


def _read_binary_operands(
    operand_values: OperandValues, state: State
) -> tuple[ChangeRecord, int, int, int]:
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    rs1_val = read_gpr(changes, state, rs1_idx)
    rs2_val = read_gpr(changes, state, rs2_idx)
    return changes, rd, rs1_val, rs2_val


def _trunc_div_signed(dividend: int, divisor: int) -> int:
    quotient = abs(dividend) // abs(divisor)
    if (dividend < 0) ^ (divisor < 0):
        return -quotient
    return quotient


def execute_mul(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute MUL: rd = (rs1 * rs2)[63:0]."""
    changes, rd, rs1_val, rs2_val = _read_binary_operands(operand_values, state)
    result = (rs1_val * rs2_val) & _MASK_64
    write_gpr(changes, state, rd, result)
    return changes


def execute_mulh(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute MULH: upper 64 bits of signed(rs1) * signed(rs2)."""
    changes, rd, rs1_val, rs2_val = _read_binary_operands(operand_values, state)
    product = _signed_64(rs1_val) * _signed_64(rs2_val)
    result = (product >> 64) & _MASK_64
    write_gpr(changes, state, rd, result)
    return changes


def execute_mulhsu(
    operand_values: OperandValues, state: State, pc: int
) -> ChangeRecord:
    """Execute MULHSU: upper 64 bits of signed(rs1) * unsigned(rs2)."""
    changes, rd, rs1_val, rs2_val = _read_binary_operands(operand_values, state)
    product = _signed_64(rs1_val) * (rs2_val & _MASK_64)
    result = (product >> 64) & _MASK_64
    write_gpr(changes, state, rd, result)
    return changes


def execute_mulhu(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute MULHU: upper 64 bits of unsigned(rs1) * unsigned(rs2)."""
    changes, rd, rs1_val, rs2_val = _read_binary_operands(operand_values, state)
    product = (rs1_val & _MASK_64) * (rs2_val & _MASK_64)
    result = (product >> 64) & _MASK_64
    write_gpr(changes, state, rd, result)
    return changes


def execute_mulw(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute MULW: rd = sign_extend((rs1[31:0] * rs2[31:0])[31:0])."""
    changes, rd, rs1_val, rs2_val = _read_binary_operands(operand_values, state)
    product_32 = (_signed_32(rs1_val) * _signed_32(rs2_val)) & _MASK_32
    write_gpr(changes, state, rd, sext32(product_32))
    return changes


def execute_div(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute DIV with RISC-V divide-by-zero and overflow semantics."""
    changes, rd, rs1_val, rs2_val = _read_binary_operands(operand_values, state)
    dividend = _signed_64(rs1_val)
    divisor = _signed_64(rs2_val)

    if divisor == 0:
        result = _MASK_64
    elif dividend == _INT_64_MIN and divisor == -1:
        result = _SIGN_64
    else:
        result = _trunc_div_signed(dividend, divisor) & _MASK_64

    write_gpr(changes, state, rd, result)
    return changes


def execute_divu(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute DIVU with RISC-V divide-by-zero semantics."""
    changes, rd, rs1_val, rs2_val = _read_binary_operands(operand_values, state)
    dividend = rs1_val & _MASK_64
    divisor = rs2_val & _MASK_64

    if divisor == 0:
        result = _MASK_64
    else:
        result = (dividend // divisor) & _MASK_64

    write_gpr(changes, state, rd, result)
    return changes


def execute_rem(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute REM with trunc-toward-zero signed remainder semantics."""
    changes, rd, rs1_val, rs2_val = _read_binary_operands(operand_values, state)
    dividend = _signed_64(rs1_val)
    divisor = _signed_64(rs2_val)

    if divisor == 0:
        result = dividend & _MASK_64
    elif dividend == _INT_64_MIN and divisor == -1:
        result = 0
    else:
        quotient = _trunc_div_signed(dividend, divisor)
        result = (dividend - (quotient * divisor)) & _MASK_64

    write_gpr(changes, state, rd, result)
    return changes


def execute_remu(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute REMU with RISC-V divide-by-zero semantics."""
    changes, rd, rs1_val, rs2_val = _read_binary_operands(operand_values, state)
    dividend = rs1_val & _MASK_64
    divisor = rs2_val & _MASK_64

    if divisor == 0:
        result = dividend
    else:
        result = dividend % divisor

    write_gpr(changes, state, rd, result & _MASK_64)
    return changes


def execute_divw(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute DIVW and sign-extend the 32-bit quotient."""
    changes, rd, rs1_val, rs2_val = _read_binary_operands(operand_values, state)
    dividend = _signed_32(rs1_val)
    divisor = _signed_32(rs2_val)

    if divisor == 0:
        quotient_32 = _MASK_32
    elif dividend == _INT_32_MIN and divisor == -1:
        quotient_32 = _SIGN_32
    else:
        quotient_32 = _trunc_div_signed(dividend, divisor) & _MASK_32

    write_gpr(changes, state, rd, sext32(quotient_32))
    return changes


def execute_divuw(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute DIVUW and sign-extend the 32-bit quotient."""
    changes, rd, rs1_val, rs2_val = _read_binary_operands(operand_values, state)
    dividend = rs1_val & _MASK_32
    divisor = rs2_val & _MASK_32

    if divisor == 0:
        quotient_32 = _MASK_32
    else:
        quotient_32 = (dividend // divisor) & _MASK_32

    write_gpr(changes, state, rd, sext32(quotient_32))
    return changes


def execute_remw(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute REMW and sign-extend the 32-bit remainder."""
    changes, rd, rs1_val, rs2_val = _read_binary_operands(operand_values, state)
    dividend = _signed_32(rs1_val)
    divisor = _signed_32(rs2_val)

    if divisor == 0:
        remainder_32 = dividend & _MASK_32
    elif dividend == _INT_32_MIN and divisor == -1:
        remainder_32 = 0
    else:
        quotient = _trunc_div_signed(dividend, divisor)
        remainder_32 = (dividend - (quotient * divisor)) & _MASK_32

    write_gpr(changes, state, rd, sext32(remainder_32))
    return changes


def execute_remuw(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute REMUW and sign-extend the 32-bit remainder."""
    changes, rd, rs1_val, rs2_val = _read_binary_operands(operand_values, state)
    dividend = rs1_val & _MASK_32
    divisor = rs2_val & _MASK_32

    if divisor == 0:
        remainder_32 = dividend
    else:
        remainder_32 = dividend % divisor

    write_gpr(changes, state, rd, sext32(remainder_32 & _MASK_32))
    return changes
