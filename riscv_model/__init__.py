# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""RISC-V functional model for instruction execution, speculation, and change tracking.

Quick start::

    from eumos import Eumos
    from riscv_model import RISCVModel

    isa   = Eumos()
    model = RISCVModel(isa)
    model.poke_gpr(1, 42)
    print(model.get_gpr(1))      # 42
"""

from riscv_model.changes import (
    BranchInfo,
    ChangeQuery,
    ChangeRecord,
    CSRWrite,
    GPRWrite,
    MemoryAccess,
)
from riscv_model.model import RISCVModel
from riscv_model.state import State

__all__ = [
    "RISCVModel",
    "State",
    "ChangeRecord",
    "ChangeQuery",
    "GPRWrite",
    "CSRWrite",
    "MemoryAccess",
    "BranchInfo",
]
