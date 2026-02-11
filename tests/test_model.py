# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Tests for RISCVModel."""

from riscv_model import RISCVModel


def test_model_initialization():
    """Test model initialization."""
    model = RISCVModel()
    assert model.get_pc() == 0
    assert model.get_gpr(0) == 0  # x0 is always zero
    assert model.get_gpr(1) == 0  # Initial value


def test_execute_addi():
    """Test executing ADDI instruction."""
    model = RISCVModel()
    # addi x1, x0, 42
    addi_instr = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (42 << 20)
    changes = model.execute(addi_instr)
    assert changes is not None
    assert model.get_gpr(1) == 42
    assert model.get_pc() == 4  # PC advances by 4


def test_execute_add():
    """Test executing ADD instruction."""
    model = RISCVModel()
    # Set up: x1 = 10, x2 = 20
    # Use internal state access for test setup (acceptable in tests)
    model._state.set_gpr(1, 10)
    model._state.set_gpr(2, 20)
    # add x3, x1, x2
    add_instr = 0x33 | (3 << 7) | (0 << 12) | (1 << 15) | (2 << 20) | (0 << 25)
    changes = model.execute(add_instr)
    assert changes is not None
    assert model.get_gpr(3) == 30


def test_speculate():
    """Test speculation mode."""
    model = RISCVModel()
    # Use internal state access for test setup
    model._state.set_gpr(1, 10)
    # addi x2, x1, 5
    addi_instr = 0x13 | (2 << 7) | (0 << 12) | (1 << 15) | (5 << 20)
    spec_changes = model.speculate(addi_instr)
    assert spec_changes is not None
    assert spec_changes.gpr_writes[0].value == 15
    # State should be unchanged
    assert model.get_gpr(2) == 0


def test_branch_taken():
    """Test branch instruction (taken)."""
    model = RISCVModel()
    # Use internal state access for test setup
    model._state.set_gpr(1, 10)
    model._state.set_gpr(2, 10)
    # beq x1, x2, 8  (branch forward 8 bytes; B-type: imm[4:1] uses shift 1, so 2 -> 4, *2 = 8)
    beq_instr = 0x63 | (0 << 12) | (1 << 15) | (2 << 20) | (2 << 8)
    changes = model.execute(beq_instr)
    assert changes is not None
    branch_info = model.get_branch_info()
    assert branch_info is not None
    assert branch_info.taken is True
    assert model.get_pc() == 8


def test_branch_not_taken():
    """Test branch instruction (not taken)."""
    model = RISCVModel()
    # Use internal state access for test setup
    model._state.set_gpr(1, 10)
    model._state.set_gpr(2, 20)
    # beq x1, x2, 8
    beq_instr = 0x63 | (0 << 12) | (1 << 15) | (2 << 20) | (8 << 7)
    changes = model.execute(beq_instr)
    assert changes is not None
    branch_info = model.get_branch_info()
    assert branch_info is not None
    assert branch_info.taken is False
    assert model.get_pc() == 4  # PC advances normally


def test_get_changes_simple_dict():
    """Test get_changes().to_simple_dict()."""
    model = RISCVModel()
    # addi x1, x0, 100
    addi_instr = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (100 << 20)
    model.execute(addi_instr)
    changes = model.get_changes()
    assert changes is not None
    d = changes.to_simple_dict()
    assert "gpr_changes" in d
    assert d["gpr_changes"][1] == 100


def test_get_changes_detailed_dict():
    """Test get_changes().to_detailed_dict()."""
    model = RISCVModel()
    # addi x1, x0, 100
    addi_instr = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (100 << 20)
    model.execute(addi_instr)
    changes = model.get_changes()
    assert changes is not None
    d = changes.to_detailed_dict()
    assert "gpr_writes" in d
    assert len(d["gpr_writes"]) == 1
    assert d["gpr_writes"][0]["register"] == 1
    assert d["gpr_writes"][0]["value"] == 100


def test_reset():
    """Test reset functionality."""
    model = RISCVModel()
    # Use internal state access for test setup
    model._state.set_gpr(1, 100)
    model._state.set_pc(0x1000)
    model.reset()
    assert model.get_gpr(1) == 0
    assert model.get_pc() == 0


def test_csr_operation():
    """Test CSR operation."""
    model = RISCVModel()
    # Use internal state access for test setup
    model._state.set_gpr(1, 0x1234)
    # csrrw x2, mstatus, x1
    csrrw_instr = 0x73 | (2 << 7) | (1 << 12) | (1 << 15) | (0x300 << 20)
    changes = model.execute(csrrw_instr)
    assert changes is not None
    # x2 should contain old mstatus value
    assert model.get_gpr(2) == 0  # Initial CSR value
    # mstatus should be updated
    assert model.get_csr("mstatus") == 0x1234


def test_x0_read_only():
    """Test that x0 is read-only."""
    model = RISCVModel()
    # addi x0, x0, 100  (should not change x0)
    addi_instr = 0x13 | (0 << 7) | (0 << 12) | (0 << 15) | (100 << 20)
    changes = model.execute(addi_instr)
    assert model.get_gpr(0) == 0
    # Change should still be tracked
    assert len(changes.gpr_writes) == 1
    assert changes.gpr_writes[0].value == 100  # Write attempted but ignored
