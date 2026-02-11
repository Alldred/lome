# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Main RISC-V functional model interface.

The :class:`RISCVModel` class is the primary public API.  It wraps a
:class:`~riscv_model.state.State`, an Eumos :class:`~eumos.decoder.Decoder`,
and an instruction executor.  It provides:

* **Instruction execution** -- :meth:`~RISCVModel.execute` and
  :meth:`~RISCVModel.speculate`.
* **Architectural register access** -- :meth:`~RISCVModel.get_gpr`,
  :meth:`~RISCVModel.set_gpr`, :meth:`~RISCVModel.get_csr`,
  :meth:`~RISCVModel.set_csr`, etc.  These honour read-only flags and
  trigger CSR side-effect hooks.
* **Raw register access (peek / poke)** -- :meth:`~RISCVModel.peek_gpr`,
  :meth:`~RISCVModel.poke_gpr`, :meth:`~RISCVModel.peek_csr`,
  :meth:`~RISCVModel.poke_csr`, etc.  Useful for test setup, debugging,
  and checkpoint restore.
* **State serialisation** -- :meth:`~RISCVModel.export_state` /
  :meth:`~RISCVModel.restore_state` for JSON round-trips.

Example -- quick start::

    from eumos import load_all_gprs, load_all_csrs
    from eumos.decoder import Decoder
    from riscv_model import RISCVModel

    gprs = load_all_gprs()
    csrs = load_all_csrs()
    dec  = Decoder()

    model = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
    model.poke_gpr(1, 10)
    model.get_gpr(1)  # => 10
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from eumos import CSRDef, GPRDef
from eumos.decoder import Decoder

from riscv_model.changes import BranchInfo, ChangeRecord
from riscv_model.executor import execute_instruction
from riscv_model.state import State


class RISCVModel:
    """RISC-V functional model for instruction execution, speculation, and change tracking.

    This is the primary public interface for the model.  Create an instance,
    optionally pre-load state with :meth:`poke_gpr` / :meth:`poke_csr` /
    :meth:`poke_pc`, then call :meth:`execute` to step through instructions.

    Examples
    --------
    Create a model and execute an ``ADDI`` instruction:

    >>> # model = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
    >>> # addi x1, x0, 42
    >>> instr = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (42 << 20)
    >>> changes = model.execute(instr)
    >>> model.get_gpr(1)
    42
    >>> model.get_pc()
    4

    Speculation (dry-run):

    >>> spec = model.speculate(instr)
    >>> model.get_gpr(1)  # unchanged
    42

    JSON round-trip:

    >>> data = model.export_state()
    >>> # model2 = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
    >>> model2.restore_state(data)
    >>> model2.get_gpr(1)
    42
    """

    # ---------------------------------------------------------------- init

    def __init__(
        self,
        *,
        decoder: Decoder,
        gpr_defs: Dict[int, GPRDef],
        csr_defs: Dict[str, CSRDef],
    ) -> None:
        """Initialise the RISC-V model.

        All three arguments are **required** and must be supplied by the
        caller (typically the ISG or application entry point).  This
        ensures a single set of Eumos objects is loaded once and shared
        across all components.

        Parameters
        ----------
        decoder : Decoder
            Eumos :class:`~eumos.decoder.Decoder` for instruction decoding.
        gpr_defs : dict[int, GPRDef]
            GPR definitions (from :func:`eumos.load_all_gprs`).
        csr_defs : dict[str, CSRDef]
            CSR definitions (from :func:`eumos.load_all_csrs`).

        Examples
        --------
        >>> from eumos import load_all_gprs, load_all_csrs
        >>> from eumos.decoder import Decoder
        >>> gprs = load_all_gprs()
        >>> csrs = load_all_csrs()
        >>> dec  = Decoder()
        >>> model = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        """
        self._state: State = State(gpr_defs=gpr_defs, csr_defs=csr_defs)
        self._decoder: Decoder = decoder
        self._last_changes: Optional[ChangeRecord] = None

    # ============================================================ execute

    def execute(
        self,
        instruction_bytes: Union[int, bytes],
        speculate: bool = False,
    ) -> Optional[ChangeRecord]:
        """Execute an instruction.

        Parameters
        ----------
        instruction_bytes : int or bytes
            32-bit instruction word.  If *bytes*, the first four bytes are
            read in **little-endian** order.
        speculate : bool, optional
            If ``True``, execute without modifying state (default ``False``).

        Returns
        -------
        ChangeRecord or None
            Record of all changes (or would-be changes in speculation
            mode).  Returns ``None`` only if the raw bytes cannot be
            decoded at all.

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> # addi x1, x0, 7
        >>> instr = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (7 << 20)
        >>> changes = m.execute(instr)
        >>> m.get_gpr(1)
        7
        >>> m.get_pc()
        4
        """
        # Convert bytes to int if needed
        if isinstance(instruction_bytes, bytes):
            if len(instruction_bytes) < 4:
                return None
            word = int.from_bytes(instruction_bytes[:4], byteorder="little")
        else:
            word = instruction_bytes & 0xFFFF_FFFF

        # Decode
        pc = self._state.get_pc()
        instance = self._decoder.decode(word, pc=pc)

        # Execute
        changes = execute_instruction(instance, self._state, pc, speculate=speculate)

        # Update PC if not speculating
        if not speculate and changes:
            if changes.pc_change:
                new_pc, _ = changes.pc_change
                self._state.set_pc(new_pc)
            else:
                # Normal sequential advance
                self._state.set_pc(pc + 4)

        self._last_changes = changes
        return changes

    def speculate(self, instruction_bytes: Union[int, bytes]) -> Optional[ChangeRecord]:
        """Execute an instruction in speculation mode (no state changes).

        This is a convenience wrapper around ``execute(..., speculate=True)``.

        Parameters
        ----------
        instruction_bytes : int or bytes
            32-bit instruction word.

        Returns
        -------
        ChangeRecord or None
            Record showing what **would** change.

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.poke_gpr(1, 10)
        0
        >>> # addi x2, x1, 5
        >>> instr = 0x13 | (2 << 7) | (0 << 12) | (1 << 15) | (5 << 20)
        >>> spec = m.speculate(instr)
        >>> spec.gpr_writes[0].value
        15
        >>> m.get_gpr(2)  # state unchanged
        0
        """
        return self.execute(instruction_bytes, speculate=True)

    # ============================================================ GPR access

    def get_gpr(self, reg: int) -> int:
        """Read a GPR (architectural -- x0 always returns 0).

        Parameters
        ----------
        reg : int
            Register index (0-31).

        Returns
        -------
        int

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.get_gpr(0)
        0
        """
        return self._state.get_gpr(reg)

    def set_gpr(self, reg: int, value: int) -> int:
        """Write a GPR (architectural -- x0 writes are ignored).

        Parameters
        ----------
        reg : int
            Register index (0-31).
        value : int
            Value to write (masked to 64 bits).

        Returns
        -------
        int
            Previous architectural value.

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.set_gpr(1, 42)
        0
        >>> m.get_gpr(1)
        42
        """
        return self._state.set_gpr(reg, value)

    def peek_gpr(self, reg: int) -> int:
        """Read raw GPR value (no x0 enforcement).

        Parameters
        ----------
        reg : int
            Register index (0-31).

        Returns
        -------
        int

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.peek_gpr(0)
        0
        """
        return self._state.peek_gpr(reg)

    def poke_gpr(self, reg: int, value: int) -> int:
        """Write raw GPR value (no x0 check, no side effects).

        Useful for test setup and state injection.

        Parameters
        ----------
        reg : int
            Register index (0-31).
        value : int
            Value to write (masked to 64 bits).

        Returns
        -------
        int
            Previous raw value.

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.poke_gpr(1, 0xFF)
        0
        >>> m.get_gpr(1)
        255
        """
        return self._state.poke_gpr(reg, value)

    # ============================================================ CSR access

    def get_csr(self, csr: Union[int, str]) -> Optional[int]:
        """Read a CSR (architectural).

        Parameters
        ----------
        csr : int or str
            12-bit address (int) or CSR name (str, e.g. ``"mstatus"``).

        Returns
        -------
        int or None
            Value, or ``None`` if the CSR does not exist.

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.get_csr("mstatus") is not None
        True
        >>> m.get_csr(0x300) is not None
        True
        """
        if isinstance(csr, str):
            return self._state.get_csr_by_name(csr)
        return self._state.get_csr(csr)

    def set_csr(self, csr: Union[int, str], value: int) -> Optional[int]:
        """Write a CSR (architectural -- respects read-only, triggers side effects).

        Parameters
        ----------
        csr : int or str
            12-bit address (int) or CSR name (str).
        value : int
            Value to write.

        Returns
        -------
        int or None
            Previous value, or ``None`` if the CSR does not exist.

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.set_csr("mstatus", 0x1800)
        0
        >>> m.get_csr("mstatus")
        6144
        """
        if isinstance(csr, str):
            return self._state.set_csr_by_name(csr, value)
        return self._state.set_csr(csr, value)

    def peek_csr(self, csr: Union[int, str]) -> Optional[int]:
        """Read raw CSR value (no side effects).

        Parameters
        ----------
        csr : int or str
            12-bit address (int) or CSR name (str).

        Returns
        -------
        int or None

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.peek_csr(0x300) is not None
        True
        """
        if isinstance(csr, str):
            return self._state.peek_csr_by_name(csr)
        return self._state.peek_csr(csr)

    def poke_csr(self, csr: Union[int, str], value: int) -> Optional[int]:
        """Write raw CSR value (bypasses read-only, no side effects).

        Parameters
        ----------
        csr : int or str
            12-bit address (int) or CSR name (str).
        value : int
            Value to write.

        Returns
        -------
        int or None
            Previous raw value, or ``None`` if the CSR does not exist.

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.poke_csr("mstatus", 0xABCD)
        0
        >>> m.peek_csr("mstatus")
        43981
        """
        if isinstance(csr, str):
            return self._state.poke_csr_by_name(csr, value)
        return self._state.poke_csr(csr, value)

    # ============================================================ PC access

    def get_pc(self) -> int:
        """Read the program counter.

        Returns
        -------
        int

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.get_pc()
        0
        """
        return self._state.get_pc()

    def set_pc(self, value: int) -> int:
        """Write the program counter (architectural).

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
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.set_pc(0x1000)
        0
        >>> m.get_pc()
        4096
        """
        return self._state.set_pc(value)

    def peek_pc(self) -> int:
        """Read raw program counter.

        Returns
        -------
        int

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.peek_pc()
        0
        """
        return self._state.peek_pc()

    def poke_pc(self, value: int) -> int:
        """Write raw program counter (no side effects).

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
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.poke_pc(0x8000_0000)
        0
        """
        return self._state.poke_pc(value)

    # ============================================================ Eumos access

    @property
    def decoder(self) -> Decoder:
        """The Eumos :class:`~eumos.decoder.Decoder` used by this model.

        Useful for sharing with other components (e.g. an ISG).

        Returns
        -------
        Decoder
        """
        return self._decoder

    @property
    def gpr_defs(self) -> Dict[int, GPRDef]:
        """GPR definitions loaded from Eumos.

        Returns
        -------
        dict[int, GPRDef]
        """
        return self._state._gpr_defs

    @property
    def csr_defs(self) -> Dict[str, CSRDef]:
        """CSR definitions loaded from Eumos.

        Returns
        -------
        dict[str, CSRDef]
        """
        return self._state._csr_defs

    # ============================================================ changes

    def get_changes(self) -> Optional[ChangeRecord]:
        """Return the change record from the most recent :meth:`execute`.

        Returns
        -------
        ChangeRecord or None

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> # addi x1, x0, 5
        >>> instr = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (5 << 20)
        >>> _ = m.execute(instr)
        >>> changes = m.get_changes()
        >>> changes.gpr_writes[0].value
        5
        """
        return self._last_changes

    def get_branch_info(self) -> Optional[BranchInfo]:
        """Return branch information from the last execution, or ``None``.

        Returns
        -------
        BranchInfo or None

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.get_branch_info() is None
        True
        """
        if self._last_changes and self._last_changes.branch_info:
            return self._last_changes.branch_info
        return None

    # ============================================================ reset

    def reset(self) -> None:
        """Reset all state to initial values and clear last changes.

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.poke_gpr(1, 100)
        0
        >>> m.poke_pc(0x1000)
        0
        >>> m.reset()
        >>> m.get_gpr(1)
        0
        >>> m.get_pc()
        0
        """
        self._state.reset()
        self._last_changes = None

    # ============================================================ JSON export / restore

    def export_state(self) -> Dict[str, Any]:
        """Export complete model state as a JSON-serialisable ``dict``.

        See :meth:`State.export_state` for the format description.

        Returns
        -------
        dict

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.poke_gpr(1, 42)
        0
        >>> data = m.export_state()
        >>> data["gprs"]["1"]
        42
        """
        return self._state.export_state()

    def restore_state(self, data: Dict[str, Any]) -> None:
        """Restore model state from a ``dict`` (as from :meth:`export_state`).

        This is a raw restore -- no side-effect hooks are triggered.
        The last-changes record is cleared.

        Parameters
        ----------
        data : dict

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> m.poke_gpr(1, 42)
        0
        >>> data = m.export_state()
        >>> m2 = RISCVModel()
        >>> m2.restore_state(data)
        >>> m2.get_gpr(1)
        42
        """
        self._state.restore_state(data)
        self._last_changes = None

    def export_state_json(self, indent: int = 2) -> str:
        """Export state as a formatted JSON string.

        Parameters
        ----------
        indent : int, optional
            Indentation level (default 2).

        Returns
        -------
        str

        Examples
        --------
        >>> # m = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)
        >>> isinstance(m.export_state_json(), str)
        True
        """
        return self._state.export_state_json(indent=indent)

    @classmethod
    def from_json(
        cls,
        json_str: str,
        *,
        decoder: Decoder,
        gpr_defs: Dict[int, GPRDef],
        csr_defs: Dict[str, CSRDef],
    ) -> "RISCVModel":
        """Create a new model from a JSON state string.

        Parameters
        ----------
        json_str : str
            As produced by :meth:`export_state_json`.
        decoder : Decoder
            Eumos decoder.
        gpr_defs : dict[int, GPRDef]
            GPR definitions.
        csr_defs : dict[str, CSRDef]
            CSR definitions.

        Returns
        -------
        RISCVModel
        """
        import json as json_mod

        model = cls(decoder=decoder, gpr_defs=gpr_defs, csr_defs=csr_defs)
        model.restore_state(json_mod.loads(json_str))
        return model
