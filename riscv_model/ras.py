# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Optional Return Address Stack (RAS) for JAL/JALR prediction.

When enabled, pushes return addresses on JAL/JALR when the destination
register is x1 (ra) or x5 (t0), and pops on JALR when the base register
is x1 or x5. Useful for ISGs or other consumers that need to track
return addresses.
"""

from __future__ import annotations

from collections import deque
from typing import Optional


class RASModel:
    """Return Address Stack for JAL/JALR tracking.

    Push on JAL/JALR when rd is x1 or x5; pop on JALR when rs1 is x1 or x5.
    """

    def __init__(self, size: int = 16) -> None:
        """Initialise RAS with given capacity.

        Parameters
        ----------
        size : int
            Maximum number of entries (default 16).
        """
        self._size = size
        self._stack: deque[int] = deque(maxlen=size)

    def push(self, return_addr: int) -> None:
        """Push a return address onto the stack."""
        self._stack.append(return_addr)

    def pop(self) -> Optional[int]:
        """Pop and return the top return address, or None if empty."""
        return self._stack.pop() if self._stack else None

    def peek(self, index: int = 0) -> Optional[int]:
        """Peek at entry; index 0 is the most recent."""
        if 0 <= index < len(self._stack):
            return self._stack[-(index + 1)]
        return None

    def __len__(self) -> int:
        return len(self._stack)
