# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Tests for Lome -- the primary public API.

Covers initialisation, instruction execution, speculation, branch handling,
change tracking (simple and detailed dicts), reset, CSR operations, and
the x0 read-only invariant.
"""


# ====================================================================
# Initialisation
# ====================================================================


class TestModelInitialisation:
    """Tests for initial model state."""

    def test_pc_starts_at_zero(self, model):
        assert model.get_pc() == 0

    def test_x0_is_zero(self, model):
        assert model.get_gpr(0) == 0

    def test_gprs_start_at_zero(self, model):
        for i in range(32):
            assert model.get_gpr(i) == 0


# ====================================================================
# Instruction execution
# ====================================================================


class TestExecution:
    """Tests for execute() with various instructions."""

    def test_execute_addi(self, model):
        """ADDI x1, x0, 42 => x1 = 42, PC = 4."""
        addi = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (42 << 20)
        changes = model.execute(addi)
        assert changes is not None
        assert model.get_gpr(1) == 42
        assert model.get_pc() == 4

    def test_execute_add(self, model):
        """ADD x3, x1, x2 with x1=10, x2=20 => x3=30."""
        model.poke_gpr(1, 10)
        model.poke_gpr(2, 20)
        add = 0x33 | (3 << 7) | (0 << 12) | (1 << 15) | (2 << 20) | (0 << 25)
        changes = model.execute(add)
        assert changes is not None
        assert model.get_gpr(3) == 30

    def test_execute_bytes(self, model):
        """Execute from little-endian bytes."""
        addi = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (7 << 20)
        instr_bytes = addi.to_bytes(4, byteorder="little")
        changes = model.execute(instr_bytes)
        assert changes is not None
        assert model.get_gpr(1) == 7

    def test_execute_short_bytes_returns_none(self, model):
        """Less than 4 bytes => None."""
        assert model.execute(b"\x00\x00") is None

    def test_pc_advances_by_4(self, model):
        """Non-branch/jump instructions advance PC by 4."""
        addi = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (1 << 20)
        model.execute(addi)
        assert model.get_pc() == 4
        model.execute(addi)
        assert model.get_pc() == 8


# ====================================================================
# Speculation
# ====================================================================


class TestSpeculation:
    """Tests for speculate() -- dry-run execution."""

    def test_speculate_returns_changes(self, model):
        model.poke_gpr(1, 10)
        addi = 0x13 | (2 << 7) | (0 << 12) | (1 << 15) | (5 << 20)
        spec = model.speculate(addi)
        assert spec is not None
        assert spec.gpr_writes[0].value == 15

    def test_speculate_does_not_modify_state(self, model):
        model.poke_gpr(1, 10)
        addi = 0x13 | (2 << 7) | (0 << 12) | (1 << 15) | (5 << 20)
        model.speculate(addi)
        assert model.get_gpr(2) == 0  # unchanged
        assert model.get_pc() == 0  # unchanged


# ====================================================================
# Branches
# ====================================================================


class TestBranches:
    """Tests for branch instruction handling."""

    def test_branch_taken(self, model):
        """BEQ x1, x2, offset when x1 == x2 => branch taken."""
        model.poke_gpr(1, 10)
        model.poke_gpr(2, 10)
        beq = 0x63 | (0 << 12) | (1 << 15) | (2 << 20) | (2 << 8)
        changes = model.execute(beq)
        assert changes is not None
        info = model.get_branch_info()
        assert info is not None
        assert info.taken is True
        assert model.get_pc() == 8

    def test_branch_not_taken(self, model):
        """BEQ x1, x2, offset when x1 != x2 => not taken, PC+4."""
        model.poke_gpr(1, 10)
        model.poke_gpr(2, 20)
        beq = 0x63 | (0 << 12) | (1 << 15) | (2 << 20) | (8 << 7)
        changes = model.execute(beq)
        assert changes is not None
        info = model.get_branch_info()
        assert info is not None
        assert info.taken is False
        assert model.get_pc() == 4

    def test_no_branch_info_for_non_branch(self, model):
        addi = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (1 << 20)
        model.execute(addi)
        assert model.get_branch_info() is None


# ====================================================================
# Change tracking
# ====================================================================


class TestChangeTracking:
    """Tests for to_simple_dict / to_detailed_dict."""

    def test_simple_dict(self, model):
        addi = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (100 << 20)
        model.execute(addi)
        d = model.get_changes().to_simple_dict()
        assert "gpr_changes" in d
        assert d["gpr_changes"][1] == 100

    def test_detailed_dict(self, model):
        addi = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (100 << 20)
        model.execute(addi)
        d = model.get_changes().to_detailed_dict()
        assert "gpr_writes" in d
        assert len(d["gpr_writes"]) == 1
        assert d["gpr_writes"][0]["register"] == 1
        assert d["gpr_writes"][0]["value"] == 100

    def test_get_changes_returns_none_before_execution(self, model):
        assert model.get_changes() is None


# ====================================================================
# Reset
# ====================================================================


class TestReset:
    """Tests for reset()."""

    def test_reset_clears_gprs(self, model):
        model.poke_gpr(1, 100)
        model.reset()
        assert model.get_gpr(1) == 0

    def test_reset_clears_pc(self, model):
        model.poke_pc(0x1000)
        model.reset()
        assert model.get_pc() == 0

    def test_reset_clears_last_changes(self, model):
        addi = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (1 << 20)
        model.execute(addi)
        model.reset()
        assert model.get_changes() is None


# ====================================================================
# CSR operations
# ====================================================================


class TestCSROperations:
    """Tests for CSR access via the model."""

    def test_csrrw_instruction(self, model):
        model.poke_gpr(1, 0x1234)
        csrrw = 0x73 | (2 << 7) | (1 << 12) | (1 << 15) | (0x300 << 20)
        changes = model.execute(csrrw)
        assert changes is not None
        assert model.get_gpr(2) == 0  # old mstatus
        assert model.get_csr("mstatus") == 0x1234

    def test_csr_by_name(self, model):
        assert model.get_csr("mstatus") is not None

    def test_csr_by_address(self, model):
        assert model.get_csr(0x300) is not None


# ====================================================================
# x0 read-only
# ====================================================================


class TestX0ReadOnly:
    """Tests for x0 behaviour."""

    def test_x0_always_zero(self, model):
        addi = 0x13 | (0 << 7) | (0 << 12) | (0 << 15) | (100 << 20)
        changes = model.execute(addi)
        assert model.get_gpr(0) == 0
        assert len(changes.gpr_writes) == 1
        assert changes.gpr_writes[0].value == 100  # tracked but ignored

    def test_set_gpr_x0_ignored(self, model):
        model.set_gpr(0, 999)
        assert model.get_gpr(0) == 0
