# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Pluggable memory interface for load/store instructions.

When a :class:`MemoryInterface` is provided to the model, load/store
instructions read from and write to it. Otherwise they use placeholder
values (loads return 0) for tests that do not need real memory.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryInterface(Protocol):
    """Protocol for memory backing load/store instructions.

    Implement this to plug external memory (e.g. from an ISG) into
    the model. Loads and stores delegate to these methods when the
    interface is provided.
    """

    def load(self, addr: int, size: int) -> int:
        """Load *size* bytes from *addr* (little-endian).

        Parameters
        ----------
        addr : int
            Byte address.
        size : int
            Access size in bytes (1, 2, 4, or 8).

        Returns
        -------
        int
            Value read, zero-extended to 64 bits.
        """
        ...

    def store(self, addr: int, value: int, size: int) -> None:
        """Store *value* (low *size* bytes) at *addr* (little-endian).

        Parameters
        ----------
        addr : int
            Byte address.
        value : int
            Value to store; only the low ``size * 8`` bits are used.
        size : int
            Access size in bytes (1, 2, 4, or 8).
        """
        ...
