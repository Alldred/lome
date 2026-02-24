# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Shared helpers for integer instruction handlers."""

from __future__ import annotations

from lome.changes import ChangeRecord, GPRRead, GPRWrite
from lome.state import State

_MASK_32 = 0xFFFF_FFFF
_MASK_64 = 0xFFFF_FFFF_FFFF_FFFF
_SIGN_32 = 0x8000_0000
_SIGN_64 = 0x8000_0000_0000_0000


def read_gpr(changes: ChangeRecord, state: State, reg: int) -> int:
    """Read a GPR and record the read in *changes*."""
    value = state.get_gpr(reg)
    if reg is not None:
        changes.gpr_reads.append(GPRRead(register=reg, value=value))
    return value


def write_gpr(changes: ChangeRecord, state: State, reg: int, value: int) -> None:
    """Write a GPR and record the write in *changes*."""
    old_value = state.set_gpr(reg, value)
    changes.gpr_writes.append(GPRWrite(register=reg, value=value, old_value=old_value))


def sext32(value: int) -> int:
    """Sign-extend a 32-bit value to 64 bits."""
    value_32 = value & _MASK_32
    if value_32 & _SIGN_32:
        return value_32 | (~_MASK_32 & _MASK_64)
    return value_32


def signed64(value: int) -> int:
    """Interpret a 64-bit integer as signed."""
    value_64 = value & _MASK_64
    if value_64 & _SIGN_64:
        return value_64 - (1 << 64)
    return value_64
