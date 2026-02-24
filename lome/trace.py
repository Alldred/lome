# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Instruction trace API.

This module re-exports the change/trace dataclasses from ``lome.changes``
under names that better reflect their role as per-instruction traces.
"""

from __future__ import annotations

from lome.changes import (
    BranchInfo,
    ChangeQuery,
    ChangeRecord,
    CSRRead,
    CSRWrite,
    FPRRead,
    FPRWrite,
    GPRRead,
    GPRWrite,
    MemoryAccess,
)

# Preferred names
InstructionTrace = ChangeRecord
TraceQuery = ChangeQuery

__all__ = [
    "InstructionTrace",
    "TraceQuery",
    "GPRRead",
    "GPRWrite",
    "FPRRead",
    "FPRWrite",
    "CSRRead",
    "CSRWrite",
    "MemoryAccess",
    "BranchInfo",
]
