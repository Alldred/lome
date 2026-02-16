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

    from eumos import Eumos
    from riscv_model import RISCVModel

    isa   = Eumos()
    model = RISCVModel(isa)
    model.poke_gpr(1, 10)
    model.get_gpr(1)  # => 10
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from eumos import CSRDef, Eumos, GPRDef

if TYPE_CHECKING:
    from riscv_model.memory import MemoryInterface
from eumos.decoder import Decoder

from riscv_model.changes import BranchInfo, ChangeRecord
from riscv_model.executor import execute_instruction
from riscv_model.memory import MemoryInterface
from riscv_model.ras import RASModel
from riscv_model.state import State


class RISCVModel:
    """RISC-V functional model for instruction execution, speculation, and change tracking.

    This is the primary public interface for the model.  Create an instance,
    optionally pre-load state with :meth:`poke_gpr` / :meth:`poke_csr` /
    :meth:`poke_pc`, then call :meth:`execute` to step through instructions.

    Examples
    --------
    Create a model and execute an ``ADDI`` instruction:

    >>> # model = RISCVModel(isa)
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
    >>> # model2 = RISCVModel(isa)
    >>> model2.restore_state(data)
    >>> model2.get_gpr(1)
    42
    """

    # ---------------------------------------------------------------- init

    def __init__(
        self,
        eumos: Eumos,
        *,
        memory: Optional[MemoryInterface] = None,
        ras: Optional[RASModel] = None,
    ) -> None:
        """Initialise the RISC-V model.

        Parameters
        ----------
        eumos : Eumos
            Shared :class:`~eumos.Eumos` ISA instance.  The model
            extracts GPR/CSR definitions and builds a
            :class:`~eumos.decoder.Decoder` from the instruction set.
        memory : MemoryInterface or None, optional
            If provided, load/store instructions read/write through it.
        ras : RASModel or None, optional
            If provided, JAL/JALR update the return address stack.

        Examples
        --------
        >>> from eumos import Eumos
        >>> model = RISCVModel(Eumos())
        """
        self._eumos: Eumos = eumos
        self._state: State = State(eumos, memory=memory)
        self._decoder: Decoder = Decoder(instructions=eumos.instructions)
        self._memory: Optional[MemoryInterface] = memory
        self._ras: Optional[RASModel] = ras
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
        >>> # m = RISCVModel(isa)
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
        instance = self._decoder.from_opc(word, pc=pc)

        # Execute
        changes = execute_instruction(
            instance,
            self._state,
            pc,
            memory=self._memory,
            ras=self._ras,
            speculate=speculate,
        )

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
        >>> # m = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
        >>> m.poke_pc(0x8000_0000)
        0
        """
        return self._state.poke_pc(value)

    # ============================================================ CSR hooks

    def register_csr_write_hook(
        self,
        csr_addr: int,
        hook: Any,
    ) -> None:
        """Register a callback invoked after an architectural CSR write.

        The hook is called as ``hook(state, address, old_value, new_value)``
        **after** the new value has been stored.  Hooks are only triggered
        by :meth:`set_csr`, **not** by :meth:`poke_csr`.

        Parameters
        ----------
        csr_addr : int
            12-bit CSR address to watch.
        hook : callable
            ``(state, address, old_value, new_value) -> None``

        Examples
        --------
        >>> # m = RISCVModel(isa)
        >>> log = []
        >>> m.register_csr_write_hook(0x300, lambda st, a, o, n: log.append((o, n)))
        >>> m.set_csr("mstatus", 0xFF)
        0
        >>> log
        [(0, 255)]
        """
        self._state.register_csr_write_hook(csr_addr, hook)

    # ============================================================ Eumos access

    @property
    def eumos(self) -> Eumos:
        """The shared :class:`~eumos.Eumos` ISA instance.

        Returns
        -------
        Eumos
        """
        return self._eumos

    @property
    def decoder(self) -> Decoder:
        """The :class:`~eumos.decoder.Decoder` used by this model.

        Returns
        -------
        Decoder
        """
        return self._decoder

    @property
    def gpr_defs(self) -> Dict[int, GPRDef]:
        """GPR definitions from the Eumos instance.

        Returns
        -------
        dict[int, GPRDef]
        """
        return self._eumos.gprs

    @property
    def csr_defs(self) -> Dict[str, CSRDef]:
        """CSR definitions from the Eumos instance.

        Returns
        -------
        dict[str, CSRDef]
        """
        return self._eumos.csrs

    # ============================================================ changes

    def get_changes(self) -> Optional[ChangeRecord]:
        """Return the change record from the most recent :meth:`execute`.

        Returns
        -------
        ChangeRecord or None

        Examples
        --------
        >>> # m = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
        >>> m.poke_gpr(1, 42)
        0
        >>> data = m.export_state()
        >>> # m2 = RISCVModel(isa)
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
        >>> # m = RISCVModel(isa)
        >>> isinstance(m.export_state_json(), str)
        True
        """
        return self._state.export_state_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str, eumos: Eumos) -> "RISCVModel":
        """Create a new model from a JSON state string.

        Parameters
        ----------
        json_str : str
            As produced by :meth:`export_state_json`.
        eumos : Eumos
            Shared Eumos ISA instance.

        Returns
        -------
        RISCVModel
        """
        import json as json_mod

        model = cls(eumos)
        model.restore_state(json_mod.loads(json_str))
        return model
