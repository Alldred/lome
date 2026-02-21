# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Floating-point helpers for RISC-V F/D extension: bits↔float, rounding mode.

Precision and rounding are implemented in a "reasonable" way using Python
float (IEEE 754). Full IEEE exception flags (fflags) and strict softfloat
semantics can be added later.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from eumos.constants import R_DYN, RNE

if TYPE_CHECKING:
    from lome.state import State

_MASK_32 = 0xFFFF_FFFF
_MASK_64 = 0xFFFF_FFFF_FFFF_FFFF

# Effective mantissa bits for "reasonable" precision (can tighten later)
PRECISION_SINGLE = 24
PRECISION_DOUBLE = 53


def bits_to_float_s(bits: int) -> float:
    """Interpret the low 32 bits as IEEE 754 single-precision; return Python float."""
    bits = int(bits) & _MASK_32
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bits_to_float_d(bits: int) -> float:
    """Interpret 64 bits as IEEE 754 double-precision; return Python float."""
    bits = int(bits) & _MASK_64
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def float_to_bits_s(value: float) -> int:
    """Convert Python float to IEEE 754 single-precision bits (low 32 bits)."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0] & _MASK_32


def float_to_bits_d(value: float) -> int:
    """Convert Python float to IEEE 754 double-precision bits (64 bits)."""
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0] & _MASK_64


def get_rounding_mode(state: "State") -> int:
    """Read dynamic rounding mode from frm CSR (bits 2:0 of fcsr). Default RNE if absent."""
    frm = state.get_csr_by_name("frm")
    if frm is None:
        return RNE
    return int(frm) & 0x7


def effective_rounding_mode(operand_values: dict, state: "State") -> int:
    """Resolve rounding mode: if instruction rm is 7 (dynamic), use frm from state; else use rm."""
    rm = operand_values.get("rm", RNE)
    if rm == R_DYN:
        return get_rounding_mode(state)
    return rm & 0x7


def round_for_float(value: float, num_bits: int, mode: int) -> float:
    """Apply RISC-V rounding mode to *value* for the given mantissa *num_bits*.

    For now we return the value unchanged; conversion via float_to_bits_s/d
    uses the platform (round-half-to-even). Strict IEEE rounding per mode
    can be added later (e.g. softfloat).
    """
    return value
