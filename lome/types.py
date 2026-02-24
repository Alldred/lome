# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Shared typing primitives for instruction execution."""

from __future__ import annotations

from typing import TypedDict


class OperandValues(TypedDict, total=False):
    """Decoded instruction operands keyed by operand name."""

    rd: int
    rs1: int
    rs2: int
    rs3: int
    imm: int
    rm: int
    pred: int
    succ: int
    fm: int
