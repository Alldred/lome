# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Shared pytest fixtures -- Eumos loaded once per session."""

from __future__ import annotations

import pytest
from eumos import Eumos

from lome import Lome
from lome.state import State

# ------------------------------------------------------------------
# Session-scoped Eumos singleton (loaded once, shared across tests)
# ------------------------------------------------------------------


@pytest.fixture(scope="session")
def eumos() -> Eumos:
    """Eumos ISA instance (session singleton)."""
    return Eumos()


# ------------------------------------------------------------------
# Convenience factories (new instance per call, shared Eumos data)
# ------------------------------------------------------------------


@pytest.fixture()
def state(eumos: Eumos) -> State:
    """Fresh :class:`State` wired to the session Eumos data."""
    return State(eumos)


@pytest.fixture()
def model(eumos: Eumos) -> Lome:
    """Fresh :class:`Lome` wired to the session Eumos data."""
    return Lome(eumos)
