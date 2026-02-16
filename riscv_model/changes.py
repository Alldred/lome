# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Change tracking for RISC-V instruction execution.

Every call to :meth:`RISCVModel.execute() <riscv_model.model.RISCVModel.execute>`
returns a :class:`ChangeRecord` that captures **all** state modifications the
instruction performed (or would have performed in speculation mode).

The record can be inspected field-by-field, queried with convenience methods
such as :meth:`ChangeRecord.get_gpr_changes`, or serialised to a ``dict``
via :meth:`ChangeRecord.to_simple_dict` / :meth:`ChangeRecord.to_detailed_dict`.

Dataclasses
-----------
* :class:`GPRWrite`  -- a single GPR write
* :class:`CSRWrite`  -- a single CSR write
* :class:`MemoryAccess` -- a load or store access
* :class:`BranchInfo` -- branch direction and target

Examples
--------
>>> from riscv_model.changes import ChangeRecord, GPRWrite
>>> cr = ChangeRecord()
>>> cr.gpr_writes.append(GPRWrite(register=1, value=42, old_value=0))
>>> cr.has_changes()
True
>>> cr.to_simple_dict()
{'gpr_changes': {1: 42}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GPRWrite:
    """Record of a single GPR write operation.

    Attributes
    ----------
    register : int
        GPR index (0-31).
    value : int
        Value written.
    old_value : int
        Value before the write.

    Examples
    --------
    >>> w = GPRWrite(register=1, value=42, old_value=0)
    >>> w.register
    1
    """

    register: int
    value: int
    old_value: int


@dataclass
class CSRWrite:
    """Record of a single CSR write operation.

    Attributes
    ----------
    address : int
        12-bit CSR address.
    name : str or None
        Human-readable CSR name (e.g. ``"mstatus"``), if known.
    value : int
        Value written.
    old_value : int
        Value before the write.

    Examples
    --------
    >>> w = CSRWrite(address=0x300, name="mstatus", value=0x1800, old_value=0)
    >>> w.name
    'mstatus'
    """

    address: int
    name: Optional[str]
    value: int
    old_value: int


@dataclass
class MemoryAccess:
    """Record of a memory access (load or store).

    Attributes
    ----------
    address : int
        Memory address.
    value : int or None
        ``None`` for loads, the stored value for stores.
    size : int
        Access size in bytes (1, 2, 4, or 8).
    is_write : bool
        ``True`` for store, ``False`` for load.

    Examples
    --------
    >>> ma = MemoryAccess(address=0x1000, value=None, size=4, is_write=False)
    >>> ma.is_write
    False
    """

    address: int
    value: Optional[int]
    size: int
    is_write: bool


@dataclass
class BranchInfo:
    """Information about a branch instruction's outcome.

    Attributes
    ----------
    taken : bool
        Whether the branch was taken.
    target : int
        Branch target address (the destination if taken, else PC+4).
    condition : str or None
        Short tag for the comparison kind (``"eq"``, ``"ne"``, ``"lt"``,
        ``"ge"``, ``"ltu"``, ``"geu"``).

    Examples
    --------
    >>> bi = BranchInfo(taken=True, target=0x100, condition="eq")
    >>> bi.taken
    True
    """

    taken: bool
    target: int
    condition: Optional[str] = None


# ---------------------------------------------------------------------------
# ChangeRecord
# ---------------------------------------------------------------------------


@dataclass
class ChangeRecord:
    """Complete record of all changes from a single instruction execution.

    Typically obtained from :meth:`RISCVModel.execute()
    <riscv_model.model.RISCVModel.execute>` or
    :meth:`RISCVModel.get_changes()
    <riscv_model.model.RISCVModel.get_changes>`.

    Attributes
    ----------
    gpr_writes : list[GPRWrite]
        GPR write operations.
    csr_writes : list[CSRWrite]
        CSR write operations.
    pc_change : tuple[int, int] or None
        ``(new_pc, old_pc)`` if the PC changed explicitly (branches, jumps).
    memory_accesses : list[MemoryAccess]
        Load and store operations.
    branch_info : BranchInfo or None
        Branch direction and target (only for branch instructions).
    exception : str or None
        Exception description (e.g. ``"illegal_instruction: foo"``).
    exception_code : int or None
        RISC-V mcause value (0-15) when a trap occurs. When set, this is
        authoritative; ``exception`` may be derived from Eumos lookup.

    Examples
    --------
    >>> cr = ChangeRecord()
    >>> cr.has_changes()
    False
    >>> cr.gpr_writes.append(GPRWrite(register=1, value=42, old_value=0))
    >>> cr.has_changes()
    True
    >>> cr.get_gpr_changes()
    {1: (42, 0)}
    """

    gpr_writes: List[GPRWrite] = field(default_factory=list)
    csr_writes: List[CSRWrite] = field(default_factory=list)
    pc_change: Optional[Tuple[int, int]] = None  # (new_pc, old_pc)
    memory_accesses: List[MemoryAccess] = field(default_factory=list)
    branch_info: Optional[BranchInfo] = None
    exception: Optional[str] = None
    exception_code: Optional[int] = None

    # ---- predicates ---------------------------------------------------

    def has_changes(self) -> bool:
        """Return ``True`` if any field records a change.

        Examples
        --------
        >>> ChangeRecord().has_changes()
        False
        >>> cr = ChangeRecord(exception="breakpoint")
        >>> cr.has_changes()
        True
        """
        return (
            len(self.gpr_writes) > 0
            or len(self.csr_writes) > 0
            or self.pc_change is not None
            or len(self.memory_accesses) > 0
            or self.branch_info is not None
            or self.exception is not None
            or self.exception_code is not None
        )

    # ---- convenience getters ------------------------------------------

    def get_gpr_changes(self) -> Dict[int, Tuple[int, int]]:
        """Return GPR changes as ``{register: (new_value, old_value)}``.

        Examples
        --------
        >>> cr = ChangeRecord()
        >>> cr.gpr_writes.append(GPRWrite(register=5, value=10, old_value=0))
        >>> cr.get_gpr_changes()
        {5: (10, 0)}
        """
        return {w.register: (w.value, w.old_value) for w in self.gpr_writes}

    def get_csr_changes(self) -> Dict[int, Tuple[int, int]]:
        """Return CSR changes as ``{address: (new_value, old_value)}``.

        Examples
        --------
        >>> cr = ChangeRecord()
        >>> cr.csr_writes.append(CSRWrite(address=0x300, name="mstatus", value=1, old_value=0))
        >>> cr.get_csr_changes()
        {768: (1, 0)}
        """
        return {w.address: (w.value, w.old_value) for w in self.csr_writes}

    def get_pc_change(self) -> Optional[Tuple[int, int]]:
        """Return ``(new_pc, old_pc)`` or ``None``.

        Examples
        --------
        >>> cr = ChangeRecord(pc_change=(0x100, 0x0))
        >>> cr.get_pc_change()
        (256, 0)
        """
        return self.pc_change

    def get_branch_info(self) -> Optional[BranchInfo]:
        """Return branch information or ``None``.

        Examples
        --------
        >>> cr = ChangeRecord()
        >>> cr.get_branch_info() is None
        True
        """
        return self.branch_info

    def get_memory_accesses(self) -> List[MemoryAccess]:
        """Return a copy of all memory accesses.

        Examples
        --------
        >>> cr = ChangeRecord()
        >>> cr.get_memory_accesses()
        []
        """
        return self.memory_accesses.copy()

    # ---- dict serialisation -------------------------------------------

    def get_all_changes(self) -> Dict:
        """Return all changes in a detailed ``dict`` format.

        This is the same as :meth:`to_detailed_dict`.

        Returns
        -------
        dict
            Keys: ``gpr_writes``, ``csr_writes``, ``pc_change``,
            ``memory_accesses``, ``branch_info``, ``exception``.

        Examples
        --------
        >>> cr = ChangeRecord()
        >>> cr.gpr_writes.append(GPRWrite(register=1, value=42, old_value=0))
        >>> d = cr.get_all_changes()
        >>> d["gpr_writes"][0]["value"]
        42
        """
        return {
            "gpr_writes": [
                {
                    "register": w.register,
                    "value": w.value,
                    "old_value": w.old_value,
                }
                for w in self.gpr_writes
            ],
            "csr_writes": [
                {
                    "address": w.address,
                    "name": w.name,
                    "value": w.value,
                    "old_value": w.old_value,
                }
                for w in self.csr_writes
            ],
            "pc_change": self.pc_change,
            "memory_accesses": [
                {
                    "address": a.address,
                    "value": a.value,
                    "size": a.size,
                    "is_write": a.is_write,
                }
                for a in self.memory_accesses
            ],
            "branch_info": (
                {
                    "taken": self.branch_info.taken,
                    "target": self.branch_info.target,
                    "condition": self.branch_info.condition,
                }
                if self.branch_info
                else None
            ),
            "exception": self.exception,
            "exception_code": self.exception_code,
        }

    def to_simple_dict(self) -> Dict:
        """Return a compact summary (new values only, no old values).

        Useful for quick inspection or logging.

        Returns
        -------
        dict
            Only keys with actual changes are included.

        Examples
        --------
        >>> cr = ChangeRecord()
        >>> cr.gpr_writes.append(GPRWrite(register=1, value=42, old_value=0))
        >>> cr.to_simple_dict()
        {'gpr_changes': {1: 42}}
        """
        result: Dict = {}
        if self.gpr_writes:
            result["gpr_changes"] = {
                reg: new for reg, (new, _) in self.get_gpr_changes().items()
            }
        if self.csr_writes:
            result["csr_changes"] = {
                addr: new for addr, (new, _) in self.get_csr_changes().items()
            }
        if self.pc_change:
            result["pc_change"] = self.pc_change[0]  # new PC only
        if self.branch_info:
            result["branch"] = {
                "taken": self.branch_info.taken,
                "target": self.branch_info.target,
            }
        if self.memory_accesses:
            result["memory_accesses"] = len(self.memory_accesses)
        if self.exception:
            result["exception"] = self.exception
        if self.exception_code is not None:
            result["exception_code"] = self.exception_code
        return result

    def to_detailed_dict(self) -> Dict:
        """Return the full change record as a ``dict``.

        Same as :meth:`get_all_changes`.

        Returns
        -------
        dict

        Examples
        --------
        >>> cr = ChangeRecord()
        >>> cr.gpr_writes.append(GPRWrite(register=1, value=42, old_value=0))
        >>> d = cr.to_detailed_dict()
        >>> d["gpr_writes"][0]["register"]
        1
        """
        return self.get_all_changes()


# ---------------------------------------------------------------------------
# ChangeQuery (thin wrapper -- prefer ChangeRecord methods directly)
# ---------------------------------------------------------------------------


class ChangeQuery:
    """Thin query wrapper around a :class:`ChangeRecord`.

    Prefer calling ``to_simple_dict()`` / ``to_detailed_dict()`` directly
    on the :class:`ChangeRecord` -- this class exists for backward
    compatibility.

    Parameters
    ----------
    change_record : ChangeRecord
        The record to wrap.

    Examples
    --------
    >>> cr = ChangeRecord()
    >>> cq = ChangeQuery(cr)
    >>> cq.has_changes()
    False
    """

    def __init__(self, change_record: ChangeRecord) -> None:
        self._record: ChangeRecord = change_record

    def simple(self) -> Dict:
        """Return simple summary of changes.

        Returns
        -------
        dict
        """
        return self._record.to_simple_dict()

    def detailed(self) -> Dict:
        """Return detailed change record.

        Returns
        -------
        dict
        """
        return self._record.to_detailed_dict()

    def has_changes(self) -> bool:
        """Check if there are any changes.

        Returns
        -------
        bool
        """
        return self._record.has_changes()
