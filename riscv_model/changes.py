# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred. All Rights Reserved

"""Change tracking for RISC-V instruction execution."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class GPRWrite:
    """Record of a GPR write operation."""

    register: int
    value: int
    old_value: int


@dataclass
class CSRWrite:
    """Record of a CSR write operation."""

    address: int
    name: Optional[str]
    value: int
    old_value: int


@dataclass
class MemoryAccess:
    """Record of a memory access (read or write)."""

    address: int
    value: Optional[int]  # None for reads, value for writes
    size: int  # Size in bytes
    is_write: bool


@dataclass
class BranchInfo:
    """Information about a branch instruction."""

    taken: bool
    target: int
    condition: Optional[str] = None  # e.g., "eq", "ne", "lt", etc.


@dataclass
class ChangeRecord:
    """Complete record of all changes from instruction execution."""

    gpr_writes: List[GPRWrite] = field(default_factory=list)
    csr_writes: List[CSRWrite] = field(default_factory=list)
    pc_change: Optional[Tuple[int, int]] = None  # (new_pc, old_pc)
    memory_accesses: List[MemoryAccess] = field(default_factory=list)
    branch_info: Optional[BranchInfo] = None
    exception: Optional[str] = None  # Exception type if any

    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return (
            len(self.gpr_writes) > 0
            or len(self.csr_writes) > 0
            or self.pc_change is not None
            or len(self.memory_accesses) > 0
            or self.branch_info is not None
            or self.exception is not None
        )

    def get_gpr_changes(self) -> Dict[int, Tuple[int, int]]:
        """Get GPR changes as dict: register -> (new_value, old_value)."""
        return {write.register: (write.value, write.old_value) for write in self.gpr_writes}

    def get_csr_changes(self) -> Dict[int, Tuple[int, int]]:
        """Get CSR changes as dict: address -> (new_value, old_value)."""
        return {write.address: (write.value, write.old_value) for write in self.csr_writes}

    def get_pc_change(self) -> Optional[Tuple[int, int]]:
        """Get PC change as (new_pc, old_pc) or None."""
        return self.pc_change

    def get_branch_info(self) -> Optional[BranchInfo]:
        """Get branch information or None."""
        return self.branch_info

    def get_memory_accesses(self) -> List[MemoryAccess]:
        """Get all memory accesses."""
        return self.memory_accesses.copy()

    def get_all_changes(self) -> Dict:
        """Get all changes in detailed format."""
        return {
            "gpr_writes": [
                {
                    "register": write.register,
                    "value": write.value,
                    "old_value": write.old_value,
                }
                for write in self.gpr_writes
            ],
            "csr_writes": [
                {
                    "address": write.address,
                    "name": write.name,
                    "value": write.value,
                    "old_value": write.old_value,
                }
                for write in self.csr_writes
            ],
            "pc_change": self.pc_change,
            "memory_accesses": [
                {
                    "address": access.address,
                    "value": access.value,
                    "size": access.size,
                    "is_write": access.is_write,
                }
                for access in self.memory_accesses
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
        }


class ChangeQuery:
    """Interface for querying changes in simple or detailed mode."""

    def __init__(self, change_record: ChangeRecord):
        """Initialize with a change record."""
        self._record = change_record

    def simple(self) -> Dict:
        """Return simple summary of changes."""
        result = {}
        if self._record.gpr_writes:
            result["gpr_changes"] = {
                reg: new_val
                for reg, (new_val, _) in self._record.get_gpr_changes().items()
            }
        if self._record.csr_writes:
            result["csr_changes"] = {
                addr: new_val
                for addr, (new_val, _) in self._record.get_csr_changes().items()
            }
        if self._record.pc_change:
            result["pc_change"] = self._record.pc_change[0]  # new PC only
        if self._record.branch_info:
            result["branch"] = {
                "taken": self._record.branch_info.taken,
                "target": self._record.branch_info.target,
            }
        if self._record.memory_accesses:
            result["memory_accesses"] = len(self._record.memory_accesses)
        if self._record.exception:
            result["exception"] = self._record.exception
        return result

    def detailed(self) -> Dict:
        """Return detailed change record."""
        return self._record.get_all_changes()

    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return self._record.has_changes()
