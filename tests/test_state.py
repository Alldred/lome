# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Tests for the State class -- peek/poke, get/set, side effects, JSON export/restore."""

import json

import pytest

from lome.state import State

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
# FPR -- peek / poke and get / set
# ====================================================================


class TestFPRPeekPoke:
    """Raw FPR access."""

    def test_poke_and_peek(self, state):
        s = state
        old = s.poke_fpr(1, 0x40490FDB)
        assert old == 0
        assert s.peek_fpr(1) == 0x40490FDB

    def test_poke_masks_to_64_bits(self, state):
        s = state
        s.poke_fpr(1, (1 << 65))
        assert s.peek_fpr(1) == 0

    def test_peek_invalid_reg_raises(self, state):
        s = state
        with pytest.raises(ValueError, match="FPR index"):
            s.peek_fpr(32)

    def test_poke_invalid_reg_raises(self, state):
        s = state
        with pytest.raises(ValueError, match="FPR index"):
            s.poke_fpr(-1, 0)


class TestFPRGetSet:
    """Architectural FPR access."""

    def test_set_and_get(self, state):
        s = state
        old = s.set_fpr(1, 0x40490FDB)
        assert old == 0
        assert s.get_fpr(1) == 0x40490FDB


class TestFPRSnapshotRestore:
    """FPRs are included in snapshot/restore."""

    def test_snapshot_restore_fprs(self, state):
        s = state
        s.set_fpr(5, 0xDEADBEEF)
        snap = s.snapshot()
        s.set_fpr(5, 0)
        assert s.get_fpr(5) == 0
        s.restore(snap)
        assert s.get_fpr(5) == 0xDEADBEEF


class TestFPRExportRestore:
    """FPRs are included in export_state/restore_state."""

    def test_export_contains_fprs(self, state):
        s = state
        data = s.export_state()
        assert "fprs" in data
        assert "fpr_names" in data.get("metadata", {})

    def test_restore_fprs(self, state):
        s = state
        s.set_fpr(3, 0x40490FDB)
        data = s.export_state()
        s2 = State(s._eumos)
        s2.restore_state(data)
        assert s2.get_fpr(3) == 0x40490FDB


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
        """poke_csr writes even to CSRs whose access is read-only."""
        s = state
        # Temporarily mark mstatus as read-only to test the bypass
        csr_def = s._csr_by_address[0x300]
        original_access = csr_def.access
        csr_def.access = "read-only"
        try:
            s.poke_csr(0x300, 0x42)
            assert s.peek_csr(0x300) == 0x42  # poke ignores access mode
        finally:
            csr_def.access = original_access

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
        """set_csr should not modify CSRs whose access is read-only."""
        s = state
        # Temporarily mark mstatus as read-only to test enforcement
        csr_def = s._csr_by_address[0x300]
        original_access = csr_def.access
        csr_def.access = "read-only"
        try:
            s.poke_csr(0x300, 0x1234)  # seed a known value via raw write
            result = s.set_csr(0x300, 0xDEADBEEF)
            assert result == 0x1234  # returns old value
            assert s.get_csr(0x300) == 0x1234  # not modified
        finally:
            csr_def.access = original_access

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

    def test_custom_hook_propagates_write(self, state):
        """A custom hook can propagate a write from one CSR to another."""
        s = state

        # Hook: when mstatus is written, copy low byte into mscratch
        def mirror_low_byte(st, addr, old_val, new_val):
            st.poke_csr(0x340, new_val & 0xFF)  # mscratch

        s.register_csr_write_hook(0x300, mirror_low_byte)
        s.set_csr(0x300, 0xABCD)
        assert s.peek_csr(0x340) == 0xCD  # low byte mirrored

    def test_poke_does_not_fire_hooks(self, state):
        """poke_csr should NOT trigger any registered hooks."""
        s = state
        fired = []
        s.register_csr_write_hook(0x300, lambda st, a, o, n: fired.append(n))
        s.poke_csr(0x300, 0x0000_0000_0000_0002)
        assert fired == []  # no hook fired


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

    def test_reset_fprs(self, state):
        s = state
        s.set_fpr(1, 0x40490FDB)
        s.reset()
        assert s.get_fpr(1) == 0


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

    def test_restore_gprs(self, state, eumos):
        s = state
        s.set_gpr(1, 42)
        data = s.export_state()
        s2 = State(eumos)
        s2.restore_state(data)
        assert s2.get_gpr(1) == 42

    def test_restore_pc(self, state, eumos):
        s = state
        s.set_pc(0x3000)
        data = s.export_state()
        s2 = State(eumos)
        s2.restore_state(data)
        assert s2.get_pc() == 0x3000

    def test_restore_csrs(self, state, eumos):
        s = state
        s.set_csr(0x300, 0x5678)
        data = s.export_state()
        s2 = State(eumos)
        s2.restore_state(data)
        assert s2.get_csr(0x300) == 0x5678

    def test_round_trip(self, state, eumos):
        """Full export -> restore round-trip preserves all state."""
        s = state
        s.set_gpr(1, 111)
        s.set_gpr(31, 222)
        s.set_pc(0x4000)
        s.set_csr(0x300, 0xAA)
        data = s.export_state()
        s2 = State(eumos)
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

    def test_from_json(self, state, eumos):
        s = state
        s.set_gpr(5, 123)
        s.set_pc(0x1000)
        json_str = s.export_state_json()
        s2 = State.from_json(json_str, eumos)
        assert s2.get_gpr(5) == 123
        assert s2.get_pc() == 0x1000

    def test_from_json_round_trip(self, state, eumos):
        """from_json -> export_state_json round-trip."""
        s = state
        s.set_gpr(3, 77)
        s.set_csr(0x300, 0x42)
        json1 = s.export_state_json()
        s2 = State.from_json(json1, eumos)
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


class TestReadOnlyIDCSRs:
    """Tests for read-only machine ID CSRs: mvendorid, marchid, mimpid, mhartid."""

    _ID_CSRS = {
        "mvendorid": 0xF11,
        "marchid": 0xF12,
        "mimpid": 0xF13,
        "mhartid": 0xF14,
    }

    def test_id_csrs_exist(self, state):
        """All four ID CSRs are present and readable."""
        s = state
        for name, addr in self._ID_CSRS.items():
            assert s.peek_csr(addr) is not None, f"{name} (0x{addr:03x}) missing"
            assert s.peek_csr_by_name(name) is not None, f"{name} not found by name"

    def test_id_csrs_reset_to_zero(self, state):
        """ID CSRs should have a reset value of zero."""
        s = state
        for name, addr in self._ID_CSRS.items():
            assert s.peek_csr(addr) == 0, f"{name} reset value should be 0"

    def test_id_csrs_are_read_only(self, state):
        """Architectural set_csr should not modify read-only ID CSRs."""
        s = state
        for name, addr in self._ID_CSRS.items():
            result = s.set_csr(addr, 0xDEAD)
            assert result == 0, f"set_csr({name}) should return old value"
            assert s.get_csr(addr) == 0, f"{name} should not be modified by set_csr"

    def test_id_csrs_read_only_by_name(self, state):
        """Architectural set_csr_by_name should not modify read-only ID CSRs."""
        s = state
        for name in self._ID_CSRS:
            result = s.set_csr_by_name(name, 0xBEEF)
            assert result == 0, f"set_csr_by_name({name}) should return old value"
            assert (
                s.get_csr_by_name(name) == 0
            ), f"{name} should not be modified by set_csr_by_name"

    def test_id_csrs_poke_bypasses_read_only(self, state):
        """poke_csr should write even though these CSRs are read-only."""
        s = state
        for name, addr in self._ID_CSRS.items():
            old = s.poke_csr(addr, 0x42)
            assert old == 0, f"poke_csr({name}) should return old value"
            assert s.peek_csr(addr) == 0x42, f"poke_csr should write {name}"

    def test_id_csrs_poke_by_name_bypasses_read_only(self, state):
        """poke_csr_by_name should write even though these CSRs are read-only."""
        s = state
        for name in self._ID_CSRS:
            old = s.poke_csr_by_name(name, 0x99)
            assert old == 0
            assert s.peek_csr_by_name(name) == 0x99

    def test_id_csrs_definitions(self, state):
        """ID CSRs should have correct definitions from Eumos."""
        s = state
        for name, addr in self._ID_CSRS.items():
            csr_def = s.get_csr_def(addr)
            assert csr_def is not None, f"{name} def not found by address"
            assert csr_def.access == "read-only", f"{name} should be read-only"
            assert csr_def.width == 64, f"{name} should be 64-bit"

            csr_def_by_name = s.get_csr_def_by_name(name)
            assert csr_def_by_name is not None, f"{name} def not found by name"
            assert csr_def_by_name.address == addr

    def test_id_csrs_in_json_export(self, state):
        """ID CSRs should appear in JSON export with their values."""
        s = state
        for name, addr in self._ID_CSRS.items():
            s.poke_csr(addr, addr)  # write a distinguishable value

        data = s.export_state()
        for name, addr in self._ID_CSRS.items():
            key = str(addr)
            assert key in data["csrs"], f"{name} missing from exported csrs"
            assert data["csrs"][key] == addr
            assert key in data["metadata"]["csr_names"]
            assert data["metadata"]["csr_names"][key] == name

    def test_id_csrs_json_round_trip(self, state, eumos):
        """ID CSRs should survive a JSON export/restore round-trip."""
        s = state
        for name, addr in self._ID_CSRS.items():
            s.poke_csr(addr, 0xCAFE + addr)

        data = s.export_state()
        s2 = State(eumos)
        s2.restore_state(data)

        for name, addr in self._ID_CSRS.items():
            assert s2.peek_csr(addr) == 0xCAFE + addr, f"{name} not restored correctly"

    def test_id_csrs_reset(self, state):
        """After reset, ID CSRs should return to their reset values (0)."""
        s = state
        for name, addr in self._ID_CSRS.items():
            s.poke_csr(addr, 0xFF)
        s.reset()
        for name, addr in self._ID_CSRS.items():
            assert s.peek_csr(addr) == 0, f"{name} not reset to 0"


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

    def test_get_fpr_def(self, state):
        s = state
        if s._fpr_defs:
            fpr_def = s.get_fpr_def(0)
            assert fpr_def is not None
            assert fpr_def.index == 0

    def test_get_fpr_def_none(self, state):
        s = state
        assert s.get_fpr_def(99) is None
