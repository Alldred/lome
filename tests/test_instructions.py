# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred. All Rights Reserved

"""Tests for instruction implementations."""

import pytest

from riscv_model import RISCVModel


def test_arithmetic_instructions():
    """Test arithmetic instructions."""
    model = RISCVModel()
    # Use internal state access for test setup
    model._state.set_gpr(1, 10)
    model._state.set_gpr(2, 20)

    # SUB: x3 = x1 - x2
    sub_instr = 0x33 | (3 << 7) | (0 << 12) | (1 << 15) | (2 << 20) | (0x20 << 25)
    model.execute(sub_instr)
    assert model.get_gpr(3) == (10 - 20) & 0xFFFFFFFFFFFFFFFF

    # ADDW: x4 = sign_extend((x1 + x2)[31:0])
    addw_instr = 0x3B | (4 << 7) | (0 << 12) | (1 << 15) | (2 << 20) | (0 << 25)
    model.execute(addw_instr)
    result = model.get_gpr(4)
    assert result == 30  # 10 + 20 = 30, sign-extended


def test_logical_instructions():
    """Test logical instructions."""
    model = RISCVModel()
    model._state.set_gpr(1, 0b1010)
    model._state.set_gpr(2, 0b1100)

    # AND: x3 = x1 & x2
    and_instr = 0x33 | (3 << 7) | (7 << 12) | (1 << 15) | (2 << 20) | (0 << 25)
    model.execute(and_instr)
    assert model.get_gpr(3) == 0b1000

    # OR: x4 = x1 | x2
    or_instr = 0x33 | (4 << 7) | (6 << 12) | (1 << 15) | (2 << 20) | (0 << 25)
    model.execute(or_instr)
    assert model.get_gpr(4) == 0b1110

    # XOR: x5 = x1 ^ x2
    xor_instr = 0x33 | (5 << 7) | (4 << 12) | (1 << 15) | (2 << 20) | (0 << 25)
    model.execute(xor_instr)
    assert model.get_gpr(5) == 0b0110


def test_shift_instructions():
    """Test shift instructions."""
    model = RISCVModel()
    model._state.set_gpr(1, 0x12345678)

    # SLLI: x2 = x1 << 4
    slli_instr = 0x13 | (2 << 7) | (1 << 12) | (1 << 15) | (4 << 20)
    model.execute(slli_instr)
    assert model.get_gpr(2) == (0x12345678 << 4) & 0xFFFFFFFFFFFFFFFF

    # SRLI: x3 = x1 >> 4
    srli_instr = 0x13 | (3 << 7) | (5 << 12) | (1 << 15) | (4 << 20)
    model.execute(srli_instr)
    assert model.get_gpr(3) == (0x12345678 >> 4) & 0xFFFFFFFFFFFFFFFF


def test_compare_instructions():
    """Test compare instructions."""
    model = RISCVModel()
    model._state.set_gpr(1, 10)
    model._state.set_gpr(2, 20)

    # SLT: x3 = (x1 < x2) ? 1 : 0
    slt_instr = 0x33 | (3 << 7) | (2 << 12) | (1 << 15) | (2 << 20) | (0 << 25)
    model.execute(slt_instr)
    assert model.get_gpr(3) == 1

    # SLTU: x4 = (x1 < x2) ? 1 : 0 (unsigned)
    sltu_instr = 0x33 | (4 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | (0 << 25)
    model.execute(sltu_instr)
    assert model.get_gpr(4) == 1


def test_jump_instructions():
    """Test jump instructions."""
    model = RISCVModel()
    model._state.set_pc(0x1000)
    model._state.set_gpr(1, 0x2000)

    # JAL: x2 = pc + 4; pc = pc + imm (eumos decoder outputs byte offset)
    # J-type: encode so decoded imm = 0x100; use imm[10:1] = 0x40 -> value 0x80, *2 = 0x100
    jal_instr = 0x6F | (2 << 7) | (0x40 << 21)
    changes = model.execute(jal_instr)
    assert model.get_gpr(2) == 0x1004  # return address
    assert model.get_pc() == 0x1100  # pc + 0x100

    # JALR: x3 = pc + 4; pc = (x1 + imm) & ~1
    model._state.set_pc(0x1000)
    jalr_instr = 0x67 | (3 << 7) | (0 << 12) | (1 << 15) | (0 << 20)
    changes = model.execute(jalr_instr)
    assert model.get_gpr(3) == 0x1004  # return address
    assert model.get_pc() == 0x2000  # (x1 + 0) & ~1


def test_system_instructions():
    """Test system instructions."""
    model = RISCVModel()

    # LUI: x1 = imm << 12
    lui_instr = 0x37 | (1 << 7) | (0x12345 << 12)
    model.execute(lui_instr)
    assert model.get_gpr(1) == (0x12345 << 12) & 0xFFFFFFFFFFFFFFFF

    # AUIPC: x2 = pc + (imm << 12)
    model._state.set_pc(0x1000)
    auipc_instr = 0x17 | (2 << 7) | (0x10 << 12)
    model.execute(auipc_instr)
    assert model.get_gpr(2) == 0x1000 + (0x10 << 12)


def test_csr_instructions():
    """Test CSR instructions."""
    model = RISCVModel()
    model._state.set_gpr(1, 0xABCD)
    model._state.set_csr(0x300, 0x1234)  # mstatus

    # CSRRS: x2 = mstatus; mstatus = mstatus | x1
    csrrs_instr = 0x73 | (2 << 7) | (2 << 12) | (1 << 15) | (0x300 << 20)
    changes = model.execute(csrrs_instr)
    assert model.get_gpr(2) == 0x1234  # old value
    assert model.get_csr(0x300) == (0x1234 | 0xABCD)

    # CSRRC: x3 = mstatus; mstatus = mstatus & ~x1
    model._state.set_gpr(1, 0x00FF)
    csrrc_instr = 0x73 | (3 << 7) | (3 << 12) | (1 << 15) | (0x300 << 20)
    changes = model.execute(csrrc_instr)
    old_val = model.get_csr(0x300)
    assert model.get_csr(0x300) == (old_val & ~0x00FF)


def test_speculation_preserves_state():
    """Test that speculation doesn't modify state."""
    model = RISCVModel()
    model._state.set_gpr(1, 10)
    model._state.set_gpr(2, 20)
    initial_pc = model.get_pc()

    # Speculate on ADD
    add_instr = 0x33 | (3 << 7) | (0 << 12) | (1 << 15) | (2 << 20) | (0 << 25)
    spec_changes = model.speculate(add_instr)

    # State should be unchanged
    assert model.get_gpr(3) == 0
    assert model.get_pc() == initial_pc
    # But changes should show what would happen
    assert spec_changes.gpr_writes[0].value == 30
