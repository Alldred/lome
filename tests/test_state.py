# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Tests for the State class -- peek/poke, get/set, side effects, JSON export/restore."""

import json

import pytest

from riscv_model.state import State

# ====================================================================
# GPR -- peek / poke (raw access)
# ====================================================================


class TestGPRPeekPoke:
    """Raw GPR access with no architectural checks."""

    def test_poke_and_peek(self, state):
        s = state
        old = s.poke_gpr(1, 42)
        assert old == 0
        assert s.peek_gpr(1) == 42

    def test_poke_x0_stores_value(self, state):
        """poke_gpr bypasses x0 read-only -- value is stored internally."""
        s = state
        s.poke_gpr(0, 0xDEAD)
        assert s.peek_gpr(0) == 0xDEAD

    def test_peek_x0_returns_stored(self, state):
        """peek_gpr returns raw storage, even for x0."""
        s = state
        s.poke_gpr(0, 99)
        assert s.peek_gpr(0) == 99

    def test_poke_masks_to_64_bits(self, state):
        s = state
        s.poke_gpr(1, (1 << 65))
        assert s.peek_gpr(1) == 0  # bit 65 masked out

    def test_poke_returns_old_value(self, state):
        s = state
        s.poke_gpr(1, 10)
        old = s.poke_gpr(1, 20)
        assert old == 10
        assert s.peek_gpr(1) == 20

    def test_peek_invalid_reg_raises(self, state):
        s = state
        with pytest.raises(ValueError, match="GPR index"):
            s.peek_gpr(32)

    def test_poke_invalid_reg_raises(self, state):
        s = state
        with pytest.raises(ValueError, match="GPR index"):
            s.poke_gpr(-1, 0)

    def test_peek_all_gprs(self, state):
        """All GPRs are accessible via peek."""
        s = state
        for i in range(32):
            s.poke_gpr(i, i * 100)
        for i in range(32):
            assert s.peek_gpr(i) == i * 100


# ====================================================================
# GPR -- get / set (architectural)
# ====================================================================


class TestGPRGetSet:
    """Architectural GPR access with x0 enforcement."""

    def test_set_and_get(self, state):
        s = state
        old = s.set_gpr(1, 42)
        assert old == 0
        assert s.get_gpr(1) == 42

    def test_x0_get_always_zero(self, state):
        s = state
        assert s.get_gpr(0) == 0

    def test_x0_set_ignored(self, state):
        s = state
        old = s.set_gpr(0, 999)
        assert old == 0
        assert s.get_gpr(0) == 0

    def test_x0_set_does_not_store(self, state):
        """set_gpr(0, x) does not modify internal storage."""
        s = state
        s.set_gpr(0, 999)
        # The internal storage for x0 should remain at reset value (0)
        assert s.peek_gpr(0) == 0

    def test_set_masks_to_64_bits(self, state):
        s = state
        s.set_gpr(1, (1 << 64) + 5)
        assert s.get_gpr(1) == 5

    def test_get_invalid_raises(self, state):
        s = state
        with pytest.raises(ValueError):
            s.get_gpr(32)

    def test_set_invalid_raises(self, state):
        s = state
        with pytest.raises(ValueError):
            s.set_gpr(-1, 0)


# ====================================================================
# CSR -- peek / poke (raw access)
# ====================================================================


class TestCSRPeekPoke:
    """Raw CSR access -- bypasses read-only and side effects."""

    def test_poke_and_peek(self, state):
        s = state
        old = s.poke_csr(0x300, 0xABCD)
        assert old == 0
        assert s.peek_csr(0x300) == 0xABCD

    def test_poke_nonexistent_returns_none(self, state):
        s = state
        assert s.poke_csr(0xFFF, 123) is None

    def test_peek_nonexistent_returns_none(self, state):
        s = state
        assert s.peek_csr(0xFFF) is None

    def test_poke_by_name(self, state):
        s = state
        old = s.poke_csr_by_name("mstatus", 0x1234)
        assert old == 0
        assert s.peek_csr_by_name("mstatus") == 0x1234

    def test_poke_by_name_nonexistent(self, state):
        s = state
        assert s.poke_csr_by_name("nonexistent_csr", 0) is None

    def test_peek_by_name_nonexistent(self, state):
        s = state
        assert s.peek_csr_by_name("nonexistent_csr") is None

    def test_poke_bypasses_read_only(self, state):
        """poke_csr writes even to read-only CSRs."""
        s = state
        # Find a read-only CSR (if any)
        for addr, csr_def in s._csr_by_address.items():
            if csr_def.access == "read-only":
                s.poke_csr(addr, 0x42)
                assert s.peek_csr(addr) == 0x42
                break

    def test_poke_does_not_trigger_hooks(self, state):
        """poke_csr should not trigger CSR write hooks."""
        s = state
        log = []
        s.register_csr_write_hook(0x300, lambda st, a, o, n: log.append(True))
        s.poke_csr(0x300, 0xFF)
        assert len(log) == 0  # no hooks fired


# ====================================================================
# CSR -- get / set (architectural)
# ====================================================================


class TestCSRGetSet:
    """Architectural CSR access with read-only enforcement and side effects."""

    def test_set_and_get(self, state):
        s = state
        old = s.set_csr(0x300, 0x1234)
        assert old == 0
        assert s.get_csr(0x300) == 0x1234

    def test_set_nonexistent_returns_none(self, state):
        s = state
        assert s.set_csr(0xFFF, 123) is None

    def test_get_nonexistent_returns_none(self, state):
        s = state
        assert s.get_csr(0xFFF) is None

    def test_set_respects_read_only(self, state):
        """set_csr should not modify read-only CSRs."""
        s = state
        for addr, csr_def in s._csr_by_address.items():
            if csr_def.access == "read-only":
                old_val = s.get_csr(addr)
                result = s.set_csr(addr, 0xDEADBEEF)
                assert result == old_val  # returns old value
                assert s.get_csr(addr) == old_val  # not modified
                break

    def test_set_by_name(self, state):
        s = state
        old = s.set_csr_by_name("mstatus", 0x5678)
        assert old == 0
        assert s.get_csr_by_name("mstatus") == 0x5678

    def test_set_by_name_nonexistent(self, state):
        s = state
        assert s.set_csr_by_name("nonexistent_csr", 0) is None

    def test_get_by_name_nonexistent(self, state):
        s = state
        assert s.get_csr_by_name("nonexistent_csr") is None

    def test_set_triggers_hooks(self, state):
        """set_csr should trigger registered write hooks."""
        s = state
        log = []
        s.register_csr_write_hook(0x300, lambda st, a, o, n: log.append((a, o, n)))
        s.set_csr(0x300, 0xFF)
        assert len(log) == 1
        assert log[0] == (0x300, 0, 0xFF)


# ====================================================================
# CSR side-effect hooks
# ====================================================================


class TestCSRSideEffects:
    """Tests for CSR write side-effect hooks."""

    def test_register_and_fire_hook(self, state):
        s = state
        fired = []
        s.register_csr_write_hook(
            0x300, lambda st, addr, old, new: fired.append((addr, old, new))
        )
        s.set_csr(0x300, 0x42)
        assert fired == [(0x300, 0, 0x42)]

    def test_multiple_hooks(self, state):
        s = state
        log1, log2 = [], []
        s.register_csr_write_hook(0x300, lambda st, a, o, n: log1.append(n))
        s.register_csr_write_hook(0x300, lambda st, a, o, n: log2.append(n))
        s.set_csr(0x300, 0x99)
        assert log1 == [0x99]
        assert log2 == [0x99]

    def test_mstatus_sstatus_mirror(self, state):
        """Writing mstatus should mirror S-mode bits to sstatus."""
        s = state
        if s.peek_csr(0x100) is None:
            pytest.skip("sstatus (0x100) not present in Eumos definitions")
        # Write a value with S-mode bits set
        s.set_csr(0x300, 0x0000_0000_0000_0002)  # SIE bit
        sstatus = s.peek_csr(0x100)
        assert sstatus is not None
        assert sstatus & 0x2 == 0x2  # SIE mirrored

    def test_sstatus_to_mstatus_mirror(self, state):
        """Writing sstatus should merge S-mode bits back into mstatus."""
        s = state
        if s.peek_csr(0x100) is None:
            pytest.skip("sstatus (0x100) not present in Eumos definitions")
        # Set mstatus to a known value first
        s.poke_csr(0x300, 0x0000_0000_0000_1800)  # MPP bits (M-mode only)
        # Write sstatus with SIE
        s.set_csr(0x100, 0x0000_0000_0000_0002)
        mstatus = s.peek_csr(0x300)
        assert mstatus is not None
        # M-mode-only bits should be preserved, SIE should be merged
        assert mstatus & 0x2 == 0x2

    def test_poke_does_not_mirror(self, state):
        """poke_csr should NOT trigger the mstatus/sstatus mirror."""
        s = state
        if s.peek_csr(0x100) is None:
            pytest.skip("sstatus (0x100) not present in Eumos definitions")
        s.poke_csr(0x300, 0x0000_0000_0000_0002)
        sstatus = s.peek_csr(0x100)
        assert sstatus == 0  # no mirroring because poke was used


# ====================================================================
# PC -- peek / poke and get / set
# ====================================================================


class TestPC:
    """Tests for PC access."""

    def test_initial_pc(self, state):
        s = state
        assert s.get_pc() == 0
        assert s.peek_pc() == 0

    def test_set_and_get_pc(self, state):
        s = state
        old = s.set_pc(0x1000)
        assert old == 0
        assert s.get_pc() == 0x1000

    def test_poke_and_peek_pc(self, state):
        s = state
        old = s.poke_pc(0x8000)
        assert old == 0
        assert s.peek_pc() == 0x8000

    def test_pc_masks_to_64_bits(self, state):
        s = state
        s.set_pc((1 << 65) + 42)
        assert s.get_pc() == 42


# ====================================================================
# Snapshot / restore
# ====================================================================


class TestSnapshot:
    """Tests for snapshot and restore (used by speculation)."""

    def test_snapshot_and_restore(self, state):
        s = state
        s.set_gpr(1, 42)
        s.set_pc(0x1000)
        snap = s.snapshot()
        s.set_gpr(1, 0)
        s.set_pc(0)
        s.restore(snap)
        assert s.get_gpr(1) == 42
        assert s.get_pc() == 0x1000

    def test_snapshot_is_a_copy(self, state):
        """Modifying state after snapshot should not affect snapshot."""
        s = state
        s.set_gpr(1, 42)
        snap = s.snapshot()
        s.set_gpr(1, 99)
        assert snap["gprs"][1] == 42

    def test_restore_is_a_copy(self, state):
        """Modifying snapshot dict after restore should not affect state."""
        s = state
        s.set_gpr(1, 42)
        snap = s.snapshot()
        s.restore(snap)
        snap["gprs"][1] = 999
        assert s.get_gpr(1) == 42


# ====================================================================
# Reset
# ====================================================================


class TestReset:
    """Tests for state reset."""

    def test_reset_gprs(self, state):
        s = state
        s.set_gpr(1, 999)
        s.reset()
        assert s.get_gpr(1) == 0

    def test_reset_pc(self, state):
        s = state
        s.set_pc(0x8000)
        s.reset()
        assert s.get_pc() == 0

    def test_reset_csrs(self, state):
        s = state
        s.set_csr(0x300, 0xDEAD)
        s.reset()
        assert s.get_csr(0x300) == 0  # back to reset value


# ====================================================================
# JSON export / restore
# ====================================================================


class TestJSONExportRestore:
    """Tests for export_state / restore_state / export_state_json / from_json."""

    def test_export_contains_expected_keys(self, state):
        s = state
        data = s.export_state()
        assert "pc" in data
        assert "gprs" in data
        assert "csrs" in data
        assert "metadata" in data

    def test_export_gprs_has_32_entries(self, state):
        s = state
        data = s.export_state()
        assert len(data["gprs"]) == 32

    def test_export_gpr_values(self, state):
        s = state
        s.set_gpr(1, 42)
        s.set_gpr(5, 100)
        data = s.export_state()
        assert data["gprs"]["1"] == 42
        assert data["gprs"]["5"] == 100

    def test_export_pc(self, state):
        s = state
        s.set_pc(0x2000)
        data = s.export_state()
        assert data["pc"] == 0x2000

    def test_export_csr_values(self, state):
        s = state
        s.set_csr(0x300, 0xBEEF)
        data = s.export_state()
        assert data["csrs"]["768"] == 0xBEEF  # 0x300 = 768

    def test_export_metadata_gpr_names(self, state):
        s = state
        data = s.export_state()
        assert "gpr_names" in data["metadata"]
        assert len(data["metadata"]["gpr_names"]) == 32

    def test_export_metadata_csr_names(self, state):
        s = state
        data = s.export_state()
        assert "csr_names" in data["metadata"]
        # mstatus should be listed
        assert "768" in data["metadata"]["csr_names"]  # 0x300

    def test_export_is_json_serialisable(self, state):
        s = state
        s.set_gpr(1, 42)
        s.set_csr(0x300, 0x1234)
        data = s.export_state()
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

    def test_restore_gprs(self, state, gpr_defs, csr_defs):
        s = state
        s.set_gpr(1, 42)
        data = s.export_state()
        s2 = State(gpr_defs=gpr_defs, csr_defs=csr_defs)
        s2.restore_state(data)
        assert s2.get_gpr(1) == 42

    def test_restore_pc(self, state, gpr_defs, csr_defs):
        s = state
        s.set_pc(0x3000)
        data = s.export_state()
        s2 = State(gpr_defs=gpr_defs, csr_defs=csr_defs)
        s2.restore_state(data)
        assert s2.get_pc() == 0x3000

    def test_restore_csrs(self, state, gpr_defs, csr_defs):
        s = state
        s.set_csr(0x300, 0x5678)
        data = s.export_state()
        s2 = State(gpr_defs=gpr_defs, csr_defs=csr_defs)
        s2.restore_state(data)
        assert s2.get_csr(0x300) == 0x5678

    def test_round_trip(self, state, gpr_defs, csr_defs):
        """Full export -> restore round-trip preserves all state."""
        s = state
        s.set_gpr(1, 111)
        s.set_gpr(31, 222)
        s.set_pc(0x4000)
        s.set_csr(0x300, 0xAA)
        data = s.export_state()
        s2 = State(gpr_defs=gpr_defs, csr_defs=csr_defs)
        s2.restore_state(data)
        assert s2.get_gpr(1) == 111
        assert s2.get_gpr(31) == 222
        assert s2.get_pc() == 0x4000
        assert s2.get_csr(0x300) == 0xAA

    def test_restore_partial_data(self, state):
        """restore_state should accept partial dicts gracefully."""
        s = state
        s.restore_state({"pc": 0x5000})
        assert s.get_pc() == 0x5000
        # GPRs and CSRs should be at their reset values
        assert s.get_gpr(1) == 0

    def test_restore_ignores_unknown_gpr_indices(self, state):
        s = state
        s.restore_state({"gprs": {"99": 123}})  # index 99 doesn't exist
        # Should not raise

    def test_restore_ignores_unknown_csr_addresses(self, state):
        s = state
        s.restore_state({"csrs": {"65535": 123}})  # 0xFFFF doesn't exist
        # Should not raise

    def test_export_state_json(self, state):
        s = state
        s.set_gpr(1, 42)
        json_str = s.export_state_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["gprs"]["1"] == 42

    def test_from_json(self, state, gpr_defs, csr_defs):
        s = state
        s.set_gpr(5, 123)
        s.set_pc(0x1000)
        json_str = s.export_state_json()
        s2 = State.from_json(json_str, gpr_defs=gpr_defs, csr_defs=csr_defs)
        assert s2.get_gpr(5) == 123
        assert s2.get_pc() == 0x1000

    def test_from_json_round_trip(self, state, gpr_defs, csr_defs):
        """from_json -> export_state_json round-trip."""
        s = state
        s.set_gpr(3, 77)
        s.set_csr(0x300, 0x42)
        json1 = s.export_state_json()
        s2 = State.from_json(json1, gpr_defs=gpr_defs, csr_defs=csr_defs)
        json2 = s2.export_state_json()
        # Parse both and compare values
        d1 = json.loads(json1)
        d2 = json.loads(json2)
        assert d1["gprs"] == d2["gprs"]
        assert d1["csrs"] == d2["csrs"]
        assert d1["pc"] == d2["pc"]


# ====================================================================
# Definition lookups
# ====================================================================


class TestDefinitionLookups:
    """Tests for get_gpr_def, get_csr_def, get_csr_def_by_name."""

    def test_get_gpr_def(self, state):
        s = state
        gpr_def = s.get_gpr_def(0)
        assert gpr_def is not None

    def test_get_gpr_def_none(self, state):
        s = state
        # Indices beyond 31 may return None (depends on Eumos)
        result = s.get_gpr_def(999)
        assert result is None

    def test_get_csr_def(self, state):
        s = state
        csr_def = s.get_csr_def(0x300)  # mstatus
        assert csr_def is not None

    def test_get_csr_def_none(self, state):
        s = state
        assert s.get_csr_def(0xFFF) is None

    def test_get_csr_def_by_name(self, state):
        s = state
        csr_def = s.get_csr_def_by_name("mstatus")
        assert csr_def is not None
        assert csr_def.address == 0x300

    def test_get_csr_def_by_name_none(self, state):
        s = state
        assert s.get_csr_def_by_name("nonexistent") is None
