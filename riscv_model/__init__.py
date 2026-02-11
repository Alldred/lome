# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""RISC-V functional model for instruction execution, speculation, and change tracking.

Quick start::

    from eumos import load_all_gprs, load_all_csrs
    from eumos.decoder import Decoder
    from riscv_model import RISCVModel

    gprs = load_all_gprs()
    csrs = load_all_csrs()
    dec  = Decoder()

    model = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
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
