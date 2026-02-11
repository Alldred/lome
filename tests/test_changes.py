# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Tests for the change-tracking dataclasses and ChangeRecord."""

from riscv_model.changes import (
    BranchInfo,
    ChangeQuery,
    ChangeRecord,
    CSRWrite,
    GPRWrite,
    MemoryAccess,
)

# ====================================================================
# Dataclass basics
# ====================================================================


class TestDataclasses:
    """Basic construction and field access for change-tracking dataclasses."""

    def test_gpr_write(self):
        w = GPRWrite(register=1, value=42, old_value=0)
        assert w.register == 1
        assert w.value == 42
        assert w.old_value == 0

    def test_csr_write(self):
        w = CSRWrite(address=0x300, name="mstatus", value=0x1800, old_value=0)
        assert w.address == 0x300
        assert w.name == "mstatus"

    def test_memory_access_load(self):
        ma = MemoryAccess(address=0x1000, value=None, size=4, is_write=False)
        assert not ma.is_write
        assert ma.value is None

    def test_memory_access_store(self):
        ma = MemoryAccess(address=0x2000, value=0xFF, size=1, is_write=True)
        assert ma.is_write
        assert ma.value == 0xFF

    def test_branch_info_taken(self):
        bi = BranchInfo(taken=True, target=0x100, condition="eq")
        assert bi.taken
        assert bi.condition == "eq"

    def test_branch_info_not_taken(self):
        bi = BranchInfo(taken=False, target=0x104)
        assert not bi.taken
        assert bi.condition is None


# ====================================================================
# ChangeRecord
# ====================================================================


class TestChangeRecord:
    """Tests for ChangeRecord methods."""

    def test_empty_has_no_changes(self):
        cr = ChangeRecord()
        assert not cr.has_changes()

    def test_has_changes_with_gpr(self):
        cr = ChangeRecord()
        cr.gpr_writes.append(GPRWrite(register=1, value=42, old_value=0))
        assert cr.has_changes()

    def test_has_changes_with_csr(self):
        cr = ChangeRecord()
        cr.csr_writes.append(
            CSRWrite(address=0x300, name="mstatus", value=1, old_value=0)
        )
        assert cr.has_changes()

    def test_has_changes_with_pc(self):
        cr = ChangeRecord(pc_change=(0x100, 0x0))
        assert cr.has_changes()

    def test_has_changes_with_memory(self):
        cr = ChangeRecord()
        cr.memory_accesses.append(
            MemoryAccess(address=0x1000, value=None, size=4, is_write=False)
        )
        assert cr.has_changes()

    def test_has_changes_with_branch(self):
        cr = ChangeRecord(branch_info=BranchInfo(taken=True, target=0x100))
        assert cr.has_changes()

    def test_has_changes_with_exception(self):
        cr = ChangeRecord(exception="breakpoint")
        assert cr.has_changes()

    def test_get_gpr_changes(self):
        cr = ChangeRecord()
        cr.gpr_writes.append(GPRWrite(register=5, value=10, old_value=0))
        cr.gpr_writes.append(GPRWrite(register=6, value=20, old_value=5))
        changes = cr.get_gpr_changes()
        assert changes == {5: (10, 0), 6: (20, 5)}

    def test_get_csr_changes(self):
        cr = ChangeRecord()
        cr.csr_writes.append(
            CSRWrite(address=0x300, name="mstatus", value=1, old_value=0)
        )
        changes = cr.get_csr_changes()
        assert changes == {0x300: (1, 0)}

    def test_get_pc_change(self):
        cr = ChangeRecord(pc_change=(0x100, 0x0))
        assert cr.get_pc_change() == (0x100, 0x0)

    def test_get_pc_change_none(self):
        cr = ChangeRecord()
        assert cr.get_pc_change() is None

    def test_get_branch_info(self):
        bi = BranchInfo(taken=True, target=0x200, condition="ne")
        cr = ChangeRecord(branch_info=bi)
        assert cr.get_branch_info() is bi

    def test_get_branch_info_none(self):
        cr = ChangeRecord()
        assert cr.get_branch_info() is None

    def test_get_memory_accesses_returns_copy(self):
        cr = ChangeRecord()
        ma = MemoryAccess(address=0x1000, value=None, size=4, is_write=False)
        cr.memory_accesses.append(ma)
        accesses = cr.get_memory_accesses()
        assert len(accesses) == 1
        # Modifying the returned list should not affect the original
        accesses.pop()
        assert len(cr.memory_accesses) == 1


# ====================================================================
# Dict serialisation
# ====================================================================


class TestDictSerialisation:
    """Tests for to_simple_dict, to_detailed_dict, get_all_changes."""

    def test_simple_dict_empty(self):
        cr = ChangeRecord()
        assert cr.to_simple_dict() == {}

    def test_simple_dict_gpr(self):
        cr = ChangeRecord()
        cr.gpr_writes.append(GPRWrite(register=1, value=42, old_value=0))
        d = cr.to_simple_dict()
        assert d == {"gpr_changes": {1: 42}}

    def test_simple_dict_csr(self):
        cr = ChangeRecord()
        cr.csr_writes.append(
            CSRWrite(address=0x300, name="mstatus", value=0xFF, old_value=0)
        )
        d = cr.to_simple_dict()
        assert d["csr_changes"] == {0x300: 0xFF}

    def test_simple_dict_pc(self):
        cr = ChangeRecord(pc_change=(0x100, 0x0))
        d = cr.to_simple_dict()
        assert d["pc_change"] == 0x100  # only new PC

    def test_simple_dict_branch(self):
        cr = ChangeRecord(
            branch_info=BranchInfo(taken=True, target=0x200, condition="eq")
        )
        d = cr.to_simple_dict()
        assert d["branch"] == {"taken": True, "target": 0x200}

    def test_simple_dict_memory(self):
        cr = ChangeRecord()
        cr.memory_accesses.append(
            MemoryAccess(address=0x1000, value=None, size=4, is_write=False)
        )
        d = cr.to_simple_dict()
        assert d["memory_accesses"] == 1

    def test_simple_dict_exception(self):
        cr = ChangeRecord(exception="breakpoint")
        d = cr.to_simple_dict()
        assert d["exception"] == "breakpoint"

    def test_detailed_dict_gpr(self):
        cr = ChangeRecord()
        cr.gpr_writes.append(GPRWrite(register=1, value=42, old_value=0))
        d = cr.to_detailed_dict()
        assert d["gpr_writes"] == [{"register": 1, "value": 42, "old_value": 0}]

    def test_detailed_dict_csr(self):
        cr = ChangeRecord()
        cr.csr_writes.append(
            CSRWrite(address=0x300, name="mstatus", value=0xFF, old_value=0)
        )
        d = cr.to_detailed_dict()
        assert d["csr_writes"][0]["name"] == "mstatus"

    def test_detailed_dict_equals_get_all_changes(self):
        cr = ChangeRecord()
        cr.gpr_writes.append(GPRWrite(register=1, value=42, old_value=0))
        assert cr.to_detailed_dict() == cr.get_all_changes()

    def test_detailed_dict_branch_none(self):
        cr = ChangeRecord()
        d = cr.to_detailed_dict()
        assert d["branch_info"] is None

    def test_detailed_dict_memory(self):
        cr = ChangeRecord()
        cr.memory_accesses.append(
            MemoryAccess(address=0x1000, value=0xFF, size=1, is_write=True)
        )
        d = cr.to_detailed_dict()
        assert d["memory_accesses"][0] == {
            "address": 0x1000,
            "value": 0xFF,
            "size": 1,
            "is_write": True,
        }


# ====================================================================
# ChangeQuery
# ====================================================================


class TestChangeQuery:
    """Tests for the ChangeQuery wrapper."""

    def test_simple(self):
        cr = ChangeRecord()
        cr.gpr_writes.append(GPRWrite(register=1, value=42, old_value=0))
        cq = ChangeQuery(cr)
        assert cq.simple() == cr.to_simple_dict()

    def test_detailed(self):
        cr = ChangeRecord()
        cq = ChangeQuery(cr)
        assert cq.detailed() == cr.to_detailed_dict()

    def test_has_changes(self):
        cr = ChangeRecord()
        cq = ChangeQuery(cr)
        assert not cq.has_changes()
        cr.exception = "test"
        assert cq.has_changes()
