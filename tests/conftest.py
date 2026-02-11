# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Shared pytest fixtures -- Eumos objects loaded once per session."""

from __future__ import annotations

from typing import Dict

import pytest
from eumos import CSRDef, GPRDef, load_all_csrs, load_all_gprs
from eumos.decoder import Decoder

from riscv_model import RISCVModel
from riscv_model.state import State

# ------------------------------------------------------------------
# Session-scoped Eumos singletons (loaded once, shared across tests)
# ------------------------------------------------------------------


@pytest.fixture(scope="session")
def gpr_defs() -> Dict[int, GPRDef]:
    """GPR definitions from Eumos (session singleton)."""
    return load_all_gprs()


@pytest.fixture(scope="session")
def csr_defs() -> Dict[str, CSRDef]:
    """CSR definitions from Eumos (session singleton)."""
    return load_all_csrs()


@pytest.fixture(scope="session")
def decoder() -> Decoder:
    """Eumos decoder (session singleton)."""
    return Decoder()


# ------------------------------------------------------------------
# Convenience factories (new instance per call, shared Eumos data)
# ------------------------------------------------------------------


@pytest.fixture()
def state(gpr_defs: Dict[int, GPRDef], csr_defs: Dict[str, CSRDef]) -> State:
    """Fresh :class:`State` wired to the session Eumos data."""
    return State(gpr_defs=gpr_defs, csr_defs=csr_defs)


@pytest.fixture()
def model(
    decoder: Decoder,
    gpr_defs: Dict[int, GPRDef],
    csr_defs: Dict[str, CSRDef],
) -> RISCVModel:
    """Fresh :class:`RISCVModel` wired to the session Eumos data."""
    return RISCVModel(decoder=decoder, gpr_defs=gpr_defs, csr_defs=csr_defs)
