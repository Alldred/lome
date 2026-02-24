# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""State management for RISC-V functional model: GPRs, CSRs, and PC.

This module provides the :class:`State` class, which holds all architectural
state for a RISC-V hart: 32 general-purpose registers (GPRs), control and
status registers (CSRs), and the program counter (PC).

Two access levels are provided for every register kind:

* **get / set** -- architectural access that enforces read-only constraints
  and triggers any side effects associated with the access (e.g. a write to
  ``mstatus`` automatically mirrors the relevant bits into ``sstatus``).
* **peek / poke** -- raw access for debugging, test setup, and state
  restore.  No side effects are triggered and read-only flags are ignored.

State can be serialised to a JSON-compatible ``dict`` with
:meth:`State.export_state` and restored with :meth:`State.restore_state`.

Example -- basic register manipulation::

    from eumos import Eumos
    from lome.state import State

    isa = Eumos()
    s = State(isa)

    s.set_gpr(1, 42)        # architectural write
    s.get_gpr(1)             # => 42
    s.peek_gpr(0)            # raw read -- x0 in storage
    s.poke_gpr(1, 99)        # raw write -- no side effects

Example -- JSON round-trip::

    data = s.export_state()
    s2 = State(isa)
    s2.restore_state(data)
    assert s2.get_gpr(1) == s.get_gpr(1)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from eumos import CSRDef, Eumos, FPRDef, GPRDef

if TYPE_CHECKING:
    from lome.memory import MemoryInterface

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MASK_64: int = 0xFFFF_FFFF_FFFF_FFFF
"""Bit-mask for 64-bit values."""

_NUM_GPRS: int = 32
"""Number of general-purpose registers."""

_NUM_FPRS: int = 32
"""Number of floating-point registers."""


# ---------------------------------------------------------------------------
# State class
# ---------------------------------------------------------------------------


class State:
    """Manages RISC-V architectural state: GPRs, CSRs, and PC.

    GPR and CSR definitions come from an :class:`~eumos.Eumos` instance
    and determine register names, widths, reset values and access modes.

    Parameters
    ----------
    eumos : Eumos
        Shared Eumos ISA instance.

    Examples
    --------
    >>> from eumos import Eumos
    >>> state = State(eumos=Eumos())
    >>> state.get_pc()
    0
    >>> state.get_gpr(0)   # x0 is hardwired to zero
    0
    """

    # ------------------------------------------------------------------ init

    def __init__(self, eumos: Eumos, memory: "MemoryInterface | None" = None) -> None:
        """Initialise state with GPRs, CSRs, and PC.

        Parameters
        ----------
        eumos : Eumos
            Shared Eumos ISA instance providing GPR and CSR definitions.
        memory : MemoryInterface or None, optional
            Optional memory backend for load/store instructions. Stored on
            state for reference; the executor passes memory directly to
            load/store instruction handlers rather than reading it from here.
        """
        self._eumos: Eumos = eumos
        self._memory: "MemoryInterface | None" = memory
        self._gpr_defs: Dict[int, GPRDef] = eumos.gprs
        self._csr_defs: Dict[str, CSRDef] = eumos.csrs
        self._csr_by_address: Dict[int, CSRDef] = {
            csr.address: csr for csr in self._csr_defs.values()
        }

        # Build reverse map: CSR address -> name (for export)
        self._csr_addr_to_name: Dict[int, str] = {
            csr.address: name for name, csr in self._csr_defs.items()
        }

        # Initialise GPRs with reset values
        self._gprs: list[int] = [
            (self._gpr_defs[idx].reset_value if idx in self._gpr_defs else 0)
            for idx in range(_NUM_GPRS)
        ]

        # Initialise CSRs with reset values
        self._csrs: Dict[int, int] = {}
        for csr_def in self._csr_defs.values():
            self._csrs[csr_def.address] = (
                csr_def.reset_value if csr_def.reset_value is not None else 0
            )

        # Initialise PC
        self._pc: int = 0

        # FPR definitions and storage (from Eumos)
        self._fpr_defs: Dict[int, FPRDef] = getattr(eumos, "fprs", {}) or {}
        self._fprs: list[int] = [
            (self._fpr_defs[idx].reset_value if idx in self._fpr_defs else 0)
            for idx in range(_NUM_FPRS)
        ]

        # CSR side-effect hooks -----------------------------------------
        # Mapping of CSR address -> list of callbacks invoked *after* an
        # architectural write via :meth:`set_csr`.  Each callback receives
        # ``(state, address, old_value, new_value)``.
        self._csr_write_hooks: Dict[int, List[Callable[..., None]]] = {}

        # Register built-in side effects
        self._register_builtin_csr_hooks()

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _mask_to_width(value: int, width: Optional[int]) -> int:
        """Mask *value* to the given bit *width* (default 64)."""
        if width is None:
            return value & _MASK_64
        if width <= 0:
            raise ValueError(f"width must be a positive integer, got {width}")
        return value & ((1 << width) - 1)

    # ====================================================================
    # GPR access
    # ====================================================================

    # ---- peek / poke (raw) ---------------------------------------------

    def peek_gpr(self, reg: int) -> int:
        """Read the raw storage value of a GPR -- no architectural checks.

        Unlike :meth:`get_gpr`, this does **not** enforce the x0=0
        constraint.  It returns whatever value is stored internally,
        which is useful for debugging and test introspection.

        Parameters
        ----------
        reg : int
            Register index (0-31).

        Returns
        -------
        int
            Raw stored value.

        Raises
        ------
        ValueError
            If *reg* is outside 0-31.

        Examples
        --------
        >>> # s = State(isa)
        >>> s.poke_gpr(0, 0xDEAD)  # force a value into x0 storage
        0
        >>> s.peek_gpr(0)           # raw read sees it
        57005
        >>> s.get_gpr(0)            # architectural read still returns 0
        0
        """
        if not (0 <= reg <= 31):
            raise ValueError(f"GPR index must be 0-31, got {reg}")
        return self._gprs[reg]

    def poke_gpr(self, reg: int, value: int) -> int:
        """Write a raw value to a GPR -- no architectural checks or side effects.

        Unlike :meth:`set_gpr`, this does **not** ignore writes to x0 and
        does **not** trigger any side effects.  The value is still masked
        to 64 bits.

        Parameters
        ----------
        reg : int
            Register index (0-31).
        value : int
            Value to write (masked to 64 bits).

        Returns
        -------
        int
            Previous raw stored value.

        Raises
        ------
        ValueError
            If *reg* is outside 0-31.

        Examples
        --------
        >>> # s = State(isa)
        >>> s.poke_gpr(1, 0xFF)
        0
        >>> s.peek_gpr(1)
        255
        """
        if not (0 <= reg <= 31):
            raise ValueError(f"GPR index must be 0-31, got {reg}")
        old = self._gprs[reg]
        self._gprs[reg] = value & _MASK_64
        return old

    # ---- get / set (architectural) -------------------------------------

    def get_gpr(self, reg: int) -> int:
        """Read a GPR with architectural semantics.

        Register x0 always returns 0 regardless of the stored value.

        Parameters
        ----------
        reg : int
            Register index (0-31).

        Returns
        -------
        int
            Architecturally visible value.

        Raises
        ------
        ValueError
            If *reg* is outside 0-31.

        Examples
        --------
        >>> # s = State(isa)
        >>> s.set_gpr(1, 42)
        0
        >>> s.get_gpr(1)
        42
        >>> s.get_gpr(0)  # always zero
        0
        """
        if not (0 <= reg <= 31):
            raise ValueError(f"GPR index must be 0-31, got {reg}")
        if reg == 0:
            return 0  # x0 is hardwired to zero
        return self._gprs[reg]

    def set_gpr(self, reg: int, value: int) -> int:
        """Write a GPR with architectural semantics.

        Writes to x0 are silently ignored (returns 0).  The value is
        masked to 64 bits.

        Parameters
        ----------
        reg : int
            Register index (0-31).
        value : int
            Value to write (masked to 64 bits).

        Returns
        -------
        int
            Previous architectural value (0 for x0).

        Raises
        ------
        ValueError
            If *reg* is outside 0-31.

        Examples
        --------
        >>> # s = State(isa)
        >>> old = s.set_gpr(1, 100)
        >>> old
        0
        >>> s.get_gpr(1)
        100
        >>> s.set_gpr(0, 999)  # silently ignored
        0
        >>> s.get_gpr(0)
        0
        """
        if not (0 <= reg <= 31):
            raise ValueError(f"GPR index must be 0-31, got {reg}")
        if reg == 0:
            return 0  # x0 is read-only
        old = self._gprs[reg]
        self._gprs[reg] = value & _MASK_64
        return old

    # ====================================================================
    # FPR access
    # ====================================================================

    # ---- peek / poke (raw) ---------------------------------------------

    def peek_fpr(self, reg: int) -> int:
        """Read the raw storage value of an FPR -- no architectural checks.

        Parameters
        ----------
        reg : int
            Register index (0-31).

        Returns
        -------
        int
            Raw stored value (64-bit).

        Raises
        ------
        ValueError
            If *reg* is outside 0-31.
        """
        if not (0 <= reg <= 31):
            raise ValueError(f"FPR index must be 0-31, got {reg}")
        return self._fprs[reg] & _MASK_64

    def poke_fpr(self, reg: int, value: int) -> int:
        """Write a raw value to an FPR -- no side effects.

        Parameters
        ----------
        reg : int
            Register index (0-31).
        value : int
            Value to write (masked to 64 bits).

        Returns
        -------
        int
            Previous raw stored value.

        Raises
        ------
        ValueError
            If *reg* is outside 0-31.
        """
        if not (0 <= reg <= 31):
            raise ValueError(f"FPR index must be 0-31, got {reg}")
        old = self._fprs[reg]
        self._fprs[reg] = value & _MASK_64
        return old

    # ---- get / set (architectural) -------------------------------------

    def get_fpr(self, reg: int) -> int:
        """Read an FPR with architectural semantics.

        Parameters
        ----------
        reg : int
            Register index (0-31).

        Returns
        -------
        int
            Value (64-bit).

        Raises
        ------
        ValueError
            If *reg* is outside 0-31.
        """
        if not (0 <= reg <= 31):
            raise ValueError(f"FPR index must be 0-31, got {reg}")
        return self._fprs[reg] & _MASK_64

    def set_fpr(self, reg: int, value: int) -> int:
        """Write an FPR with architectural semantics.

        Parameters
        ----------
        reg : int
            Register index (0-31).
        value : int
            Value to write (masked to 64 bits).

        Returns
        -------
        int
            Previous value.

        Raises
        ------
        ValueError
            If *reg* is outside 0-31.
        """
        if not (0 <= reg <= 31):
            raise ValueError(f"FPR index must be 0-31, got {reg}")
        old = self._fprs[reg]
        self._fprs[reg] = value & _MASK_64
        return old

    # ====================================================================
    # CSR access
    # ====================================================================

    # ---- peek / poke (raw) ---------------------------------------------

    def peek_csr(self, csr: int) -> Optional[int]:
        """Read the raw storage value of a CSR -- no side effects.

        Parameters
        ----------
        csr : int
            12-bit CSR address.

        Returns
        -------
        int or None
            Raw value, or ``None`` if the CSR does not exist.

        Examples
        --------
        >>> # s = State(isa)
        >>> val = s.peek_csr(0x300)  # mstatus
        >>> val is not None
        True
        """
        return self._csrs.get(csr & 0xFFF)

    def poke_csr(self, csr: int, value: int) -> Optional[int]:
        """Write a raw value to a CSR -- bypasses read-only and side effects.

        The value is still masked to the CSR's defined width (or 64 bits
        if no width is specified).

        Parameters
        ----------
        csr : int
            12-bit CSR address.
        value : int
            Value to write.

        Returns
        -------
        int or None
            Previous raw value, or ``None`` if the CSR does not exist.

        Examples
        --------
        >>> # s = State(isa)
        >>> s.poke_csr(0x300, 0xABCD)  # force mstatus
        0
        >>> s.peek_csr(0x300)
        43981
        """
        addr = csr & 0xFFF
        if addr not in self._csrs:
            return None
        old = self._csrs[addr]
        csr_def = self._csr_by_address.get(addr)
        self._csrs[addr] = self._mask_to_width(
            value, csr_def.width if csr_def else None
        )
        return old

    def peek_csr_by_name(self, name: str) -> Optional[int]:
        """Read raw CSR value by name -- no side effects.

        Parameters
        ----------
        name : str
            CSR name (e.g. ``"mstatus"``).

        Returns
        -------
        int or None
            Raw value, or ``None`` if the name is unknown.

        Examples
        --------
        >>> # s = State(isa)
        >>> s.peek_csr_by_name("mstatus") is not None
        True
        """
        csr_def = self._csr_defs.get(name)
        if csr_def is None:
            return None
        return self.peek_csr(csr_def.address)

    def poke_csr_by_name(self, name: str, value: int) -> Optional[int]:
        """Write raw CSR value by name -- bypasses read-only and side effects.

        Parameters
        ----------
        name : str
            CSR name (e.g. ``"mstatus"``).
        value : int
            Value to write.

        Returns
        -------
        int or None
            Previous raw value, or ``None`` if the name is unknown.

        Examples
        --------
        >>> # s = State(isa)
        >>> s.poke_csr_by_name("mstatus", 0x1234)
        0
        """
        csr_def = self._csr_defs.get(name)
        if csr_def is None:
            return None
        return self.poke_csr(csr_def.address, value)

    # ---- get / set (architectural) -------------------------------------

    def get_csr(self, csr: int) -> Optional[int]:
        """Read a CSR with architectural semantics.

        Parameters
        ----------
        csr : int
            12-bit CSR address.

        Returns
        -------
        int or None
            Value, or ``None`` if the CSR does not exist.

        Examples
        --------
        >>> # s = State(isa)
        >>> s.set_csr(0x300, 0x1800)  # mstatus
        0
        >>> s.get_csr(0x300)
        6144
        """
        addr = csr & 0xFFF
        return self._csrs.get(addr)

    def set_csr(self, csr: int, value: int) -> Optional[int]:
        """Write a CSR with architectural semantics.

        Respects read-only access modes: if the CSR is read-only the value
        is **not** written but the old value is still returned.  After a
        successful write any registered side-effect hooks are called (e.g.
        mirroring ``mstatus`` bits into ``sstatus``).

        Parameters
        ----------
        csr : int
            12-bit CSR address.
        value : int
            Value to write.

        Returns
        -------
        int or None
            Previous value, or ``None`` if the CSR does not exist.

        Examples
        --------
        >>> # s = State(isa)
        >>> s.set_csr(0x300, 0x1800)  # write mstatus
        0
        >>> s.get_csr(0x300)
        6144
        """
        addr = csr & 0xFFF
        if addr not in self._csrs:
            return None

        csr_def = self._csr_by_address.get(addr)

        # Read-only CSRs cannot be written architecturally
        if csr_def and csr_def.access == "read-only":
            return self._csrs.get(addr)

        old = self._csrs[addr]
        new = self._mask_to_width(value, csr_def.width if csr_def else None)
        self._csrs[addr] = new

        # Trigger side-effect hooks
        self._fire_csr_write_hooks(addr, old, new)

        return old

    def get_csr_by_name(self, name: str) -> Optional[int]:
        """Read CSR value by name with architectural semantics.

        Parameters
        ----------
        name : str
            CSR name (e.g. ``"mstatus"``).

        Returns
        -------
        int or None
            Value, or ``None`` if the name is unknown.

        Examples
        --------
        >>> # s = State(isa)
        >>> s.get_csr_by_name("mstatus") is not None
        True
        """
        csr_def = self._csr_defs.get(name)
        if csr_def is None:
            return None
        return self.get_csr(csr_def.address)

    def set_csr_by_name(self, name: str, value: int) -> Optional[int]:
        """Write CSR value by name with architectural semantics.

        Respects read-only flags and triggers side-effect hooks.

        Parameters
        ----------
        name : str
            CSR name (e.g. ``"mstatus"``).
        value : int
            Value to write.

        Returns
        -------
        int or None
            Previous value, or ``None`` if the name is unknown.

        Examples
        --------
        >>> # s = State(isa)
        >>> s.set_csr_by_name("mstatus", 0x1800)
        0
        """
        csr_def = self._csr_defs.get(name)
        if csr_def is None:
            return None
        return self.set_csr(csr_def.address, value)

    # ====================================================================
    # PC access
    # ====================================================================

    def peek_pc(self) -> int:
        """Read raw program counter value.

        Returns
        -------
        int

        Examples
        --------
        >>> # s = State(isa)
        >>> s.peek_pc()
        0
        """
        return self._pc

    def poke_pc(self, value: int) -> int:
        """Write raw program counter -- no side effects.

        Parameters
        ----------
        value : int
            New PC (masked to 64 bits).

        Returns
        -------
        int
            Previous PC value.

        Examples
        --------
        >>> # s = State(isa)
        >>> s.poke_pc(0x8000_0000)
        0
        >>> s.peek_pc()
        2147483648
        """
        old = self._pc
        self._pc = value & _MASK_64
        return old

    def get_pc(self) -> int:
        """Read the program counter.

        Returns
        -------
        int

        Examples
        --------
        >>> # s = State(isa)
        >>> s.get_pc()
        0
        """
        return self._pc

    def set_pc(self, value: int) -> int:
        """Write the program counter.

        Parameters
        ----------
        value : int
            New PC (masked to 64 bits).

        Returns
        -------
        int
            Previous PC value.

        Examples
        --------
        >>> # s = State(isa)
        >>> s.set_pc(0x1000)
        0
        >>> s.get_pc()
        4096
        """
        old = self._pc
        self._pc = value & _MASK_64
        return old

    # ====================================================================
    # CSR side-effect hooks
    # ====================================================================

    def register_csr_write_hook(
        self, csr_addr: int, hook: Callable[["State", int, int, int], None]
    ) -> None:
        """Register a callback invoked after an architectural CSR write.

        The hook is called as ``hook(state, address, old_value, new_value)``
        **after** the new value has been stored.  Multiple hooks may be
        registered for the same address and are called in registration order.

        Hooks are only triggered by :meth:`set_csr` / :meth:`set_csr_by_name`,
        **not** by :meth:`poke_csr` / :meth:`poke_csr_by_name`.

        Parameters
        ----------
        csr_addr : int
            12-bit CSR address to watch.
        hook : callable
            ``(state, address, old_value, new_value) -> None``

        Examples
        --------
        >>> # s = State(isa)
        >>> log = []
        >>> s.register_csr_write_hook(0x300, lambda st, a, o, n: log.append((a, o, n)))
        >>> s.set_csr(0x300, 0xFF)
        0
        >>> len(log)
        1
        """
        self._csr_write_hooks.setdefault(csr_addr, []).append(hook)

    def _fire_csr_write_hooks(self, addr: int, old: int, new: int) -> None:
        """Invoke all registered write hooks for *addr*."""
        for hook in self._csr_write_hooks.get(addr, []):
            hook(self, addr, old, new)

    # ---- built-in hooks ------------------------------------------------

    def _register_builtin_csr_hooks(self) -> None:
        """Register built-in CSR side-effect hooks.

        Currently registers:

        * **mstatus / sstatus mirroring** -- writing ``mstatus`` (0x300)
          mirrors the S-mode visible bits into ``sstatus`` (0x100), and
          vice-versa.  Only registered if both CSRs are present in the
          Eumos definitions.
        """
        # mstatus (0x300) <-> sstatus (0x100) mirroring
        mstatus_def = self._csr_by_address.get(0x300)
        sstatus_def = self._csr_by_address.get(0x100)
        if mstatus_def and sstatus_def:
            self.register_csr_write_hook(0x300, _mirror_mstatus_to_sstatus)
            self.register_csr_write_hook(0x100, _mirror_sstatus_to_mstatus)

    # ====================================================================
    # Definition lookups
    # ====================================================================

    def get_gpr_def(self, reg: int) -> Optional[GPRDef]:
        """Return the Eumos :class:`GPRDef` for register *reg*.

        Parameters
        ----------
        reg : int
            Register index (0-31).

        Returns
        -------
        GPRDef or None
        """
        return self._gpr_defs.get(reg)

    def get_csr_def(self, csr: int) -> Optional[CSRDef]:
        """Return the Eumos :class:`CSRDef` for CSR address *csr*.

        Parameters
        ----------
        csr : int
            12-bit CSR address.

        Returns
        -------
        CSRDef or None
        """
        return self._csr_by_address.get(csr & 0xFFF)

    def get_csr_def_by_name(self, name: str) -> Optional[CSRDef]:
        """Return the Eumos :class:`CSRDef` for CSR *name*.

        Parameters
        ----------
        name : str
            CSR name (e.g. ``"mstatus"``).

        Returns
        -------
        CSRDef or None
        """
        return self._csr_defs.get(name)

    def get_fpr_def(self, reg: int) -> Optional[FPRDef]:
        """Return the Eumos :class:`FPRDef` for FPR *reg*.

        Parameters
        ----------
        reg : int
            Register index (0-31).

        Returns
        -------
        FPRDef or None
        """
        return self._fpr_defs.get(reg)

    # ====================================================================
    # Snapshot / restore (for speculation -- raw, no hooks)
    # ====================================================================

    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of current state for speculation.

        The snapshot is a plain ``dict`` suitable for passing to
        :meth:`restore`.  This is a **raw** copy -- restoring does not
        trigger any side-effect hooks.

        Returns
        -------
        dict
            Keys: ``"gprs"``, ``"csrs"``, ``"pc"``.

        Examples
        --------
        >>> # s = State(isa)
        >>> s.set_gpr(1, 42)
        0
        >>> snap = s.snapshot()
        >>> s.set_gpr(1, 0)
        42
        >>> s.restore(snap)
        >>> s.get_gpr(1)
        42
        """
        return {
            "gprs": self._gprs.copy(),
            "csrs": self._csrs.copy(),
            "fprs": self._fprs.copy(),
            "pc": self._pc,
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore state from a snapshot (raw -- no side-effect hooks).

        Parameters
        ----------
        snapshot : dict
            As returned by :meth:`snapshot`. Keys ``gprs``, ``csrs``, ``fprs``,
            and ``pc`` are optional; only present keys are restored.
        """
        if "gprs" in snapshot:
            gprs_snapshot = snapshot["gprs"]
            if isinstance(gprs_snapshot, list):
                restored = [int(v) & _MASK_64 for v in gprs_snapshot[:_NUM_GPRS]]
                if len(restored) < _NUM_GPRS:
                    restored.extend([0] * (_NUM_GPRS - len(restored)))
                self._gprs = restored
            elif isinstance(gprs_snapshot, dict):
                restored = self._gprs.copy()
                for key, value in gprs_snapshot.items():
                    idx = int(key)
                    if 0 <= idx < _NUM_GPRS:
                        restored[idx] = int(value) & _MASK_64
                self._gprs = restored
        if "csrs" in snapshot:
            self._csrs = snapshot["csrs"].copy()
        if "fprs" in snapshot:
            fprs_snapshot = snapshot["fprs"]
            if isinstance(fprs_snapshot, list):
                restored = [int(v) & _MASK_64 for v in fprs_snapshot[:_NUM_FPRS]]
                if len(restored) < _NUM_FPRS:
                    restored.extend([0] * (_NUM_FPRS - len(restored)))
                self._fprs = restored
            elif isinstance(fprs_snapshot, dict):
                restored = self._fprs.copy()
                for key, value in fprs_snapshot.items():
                    idx = int(key)
                    if 0 <= idx < _NUM_FPRS:
                        restored[idx] = int(value) & _MASK_64
                self._fprs = restored
        if "pc" in snapshot:
            self._pc = snapshot["pc"]

    # ====================================================================
    # Reset
    # ====================================================================

    def reset(self) -> None:
        """Reset all state to initial / reset values.

        GPRs and CSRs are set to their Eumos-defined reset values, and the
        PC is set to 0.  Side-effect hooks are **not** triggered.

        Examples
        --------
        >>> # s = State(isa)
        >>> s.set_gpr(1, 999)
        0
        >>> s.set_pc(0x8000)
        0
        >>> s.reset()
        >>> s.get_gpr(1)
        0
        >>> s.get_pc()
        0
        """
        for idx in range(_NUM_GPRS):
            gpr_def = self._gpr_defs.get(idx)
            self._gprs[idx] = gpr_def.reset_value if gpr_def else 0

        for csr_def in self._csr_defs.values():
            self._csrs[csr_def.address] = (
                csr_def.reset_value if csr_def.reset_value is not None else 0
            )

        for idx in range(_NUM_FPRS):
            fpr_def = self._fpr_defs.get(idx)
            self._fprs[idx] = fpr_def.reset_value if fpr_def else 0

        self._pc = 0

    # ====================================================================
    # JSON export / restore
    # ====================================================================

    def export_state(self) -> Dict[str, Any]:
        """Export complete state as a JSON-serialisable ``dict``.

        The format is designed for both machine consumption (restore) and
        human readability.  Integer keys are stored as strings to ensure
        JSON compatibility.

        Returns
        -------
        dict
            Structure::

                {
                    "pc": <int>,
                    "gprs": { "<index>": <value>, ... },   # 32 entries
                    "csrs": { "<address>": <value>, ... },
                    "metadata": {
                        "gpr_names": { "<index>": "<name>", ... },
                        "csr_names": { "<address>": "<name>", ... }
                    }
                }

        Examples
        --------
        >>> # s = State(isa)
        >>> s.set_gpr(1, 42)
        0
        >>> data = s.export_state()
        >>> data["pc"]
        0
        >>> data["gprs"]["1"]
        42
        >>> json.dumps(data) is not None  # it's JSON-safe
        True
        """
        gprs: Dict[str, int] = {}
        gpr_names: Dict[str, str] = {}
        for idx in range(_NUM_GPRS):
            gprs[str(idx)] = self._gprs[idx]
            gpr_def = self._gpr_defs.get(idx)
            gpr_names[str(idx)] = gpr_def.abi_name if gpr_def else f"x{idx}"

        csrs: Dict[str, int] = {}
        csr_names: Dict[str, str] = {}
        for addr, value in sorted(self._csrs.items()):
            key = str(addr)
            csrs[key] = value
            name = self._csr_addr_to_name.get(addr)
            if name:
                csr_names[key] = name

        fprs: Dict[str, int] = {}
        fpr_names: Dict[str, str] = {}
        for idx in range(_NUM_FPRS):
            fprs[str(idx)] = self._fprs[idx]
            fpr_def = self._fpr_defs.get(idx)
            fpr_names[str(idx)] = fpr_def.abi_name if fpr_def else f"f{idx}"

        return {
            "pc": self._pc,
            "gprs": gprs,
            "csrs": csrs,
            "fprs": fprs,
            "metadata": {
                "gpr_names": gpr_names,
                "csr_names": csr_names,
                "fpr_names": fpr_names,
            },
        }

    def restore_state(self, data: Dict[str, Any]) -> None:
        """Restore state from a ``dict`` produced by :meth:`export_state`.

        Missing keys are silently skipped so partial state dicts are
        accepted.  Unknown GPR indices or CSR addresses are ignored.
        This is a **raw** restore -- no side-effect hooks are triggered.

        Parameters
        ----------
        data : dict
            As returned by :meth:`export_state`.

        Examples
        --------
        >>> # s = State(isa)
        >>> s.set_gpr(1, 42)
        0
        >>> data = s.export_state()
        >>> # s2 = State(isa)
        >>> s2.restore_state(data)
        >>> s2.get_gpr(1)
        42
        """
        if "pc" in data:
            self._pc = int(data["pc"]) & _MASK_64

        if "gprs" in data:
            gprs_data = data["gprs"]
            if isinstance(gprs_data, list):
                for idx, value in enumerate(gprs_data[:_NUM_GPRS]):
                    self._gprs[idx] = int(value) & _MASK_64
            else:
                for key, value in gprs_data.items():
                    idx = int(key)
                    if 0 <= idx <= 31:
                        self._gprs[idx] = int(value) & _MASK_64

        if "csrs" in data:
            for key, value in data["csrs"].items():
                addr = int(key)
                if addr in self._csrs:
                    csr_def = self._csr_by_address.get(addr)
                    self._csrs[addr] = self._mask_to_width(
                        int(value), csr_def.width if csr_def else None
                    )

        if "fprs" in data:
            fprs_data = data["fprs"]
            if isinstance(fprs_data, list):
                for idx, value in enumerate(fprs_data[:_NUM_FPRS]):
                    self._fprs[idx] = int(value) & _MASK_64
            else:
                for key, value in fprs_data.items():
                    idx = int(key)
                    if 0 <= idx <= 31:
                        self._fprs[idx] = int(value) & _MASK_64

    def export_state_json(self, indent: int = 2) -> str:
        """Export state as a formatted JSON string.

        Parameters
        ----------
        indent : int, optional
            JSON indentation level (default 2).

        Returns
        -------
        str
            JSON string.

        Examples
        --------
        >>> # s = State(isa)
        >>> isinstance(s.export_state_json(), str)
        True
        """
        return json.dumps(self.export_state(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str, eumos: Eumos) -> "State":
        """Create a new State instance from a JSON string.

        Parameters
        ----------
        json_str : str
            JSON string as produced by :meth:`export_state_json`.
        eumos : Eumos
            Shared Eumos ISA instance.

        Returns
        -------
        State
            New state with values from the JSON.
        """
        state = cls(eumos)
        state.restore_state(json.loads(json_str))
        return state


# ---------------------------------------------------------------------------
# Built-in CSR side-effect hooks
# ---------------------------------------------------------------------------

# S-mode status bits that sstatus mirrors from mstatus
# (SD, MXR, SUM, XS, FS, SPP, SPIE, UPIE, SIE, UIE, and VS bits)
_SSTATUS_MASK: int = 0x8000_0003_000D_E762


def _mirror_mstatus_to_sstatus(state: State, _addr: int, _old: int, new: int) -> None:
    """After writing mstatus, update sstatus with the S-mode visible bits."""
    if 0x100 in state._csrs:
        state._csrs[0x100] = new & _SSTATUS_MASK


def _mirror_sstatus_to_mstatus(state: State, _addr: int, _old: int, new: int) -> None:
    """After writing sstatus, merge S-mode bits back into mstatus."""
    if 0x300 in state._csrs:
        mstatus = state._csrs[0x300]
        state._csrs[0x300] = (mstatus & ~_SSTATUS_MASK) | (new & _SSTATUS_MASK)
