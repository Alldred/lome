# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""RISC-V functional model for instruction execution, speculation, and change tracking.

Quick start::

    from eumos import Eumos
    from lome import Lome

    isa   = Eumos()
    model = Lome(isa)
    model.poke_gpr(1, 42)
    print(model.get_gpr(1))      # 42
"""

from lome.memory import MemoryInterface
from lome.model import Lome
from lome.ras import RASModel
from lome.state import State
from lome.trace import (
    BranchInfo,
    CSRWrite,
    FPRWrite,
    GPRWrite,
    InstructionTrace,
    MemoryAccess,
    TraceQuery,
)

__all__ = [
    "MemoryInterface",
    "RASModel",
    "Lome",
    "State",
    "InstructionTrace",
    "TraceQuery",
    "GPRWrite",
    "FPRWrite",
    "CSRWrite",
    "MemoryAccess",
    "BranchInfo",
]
