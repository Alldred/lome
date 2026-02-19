# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Tests for floating-point instructions and FPR state."""

import struct

from riscv_model import RISCVModel
from riscv_model.instructions import float as float_ins


def _bits_s(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0] & 0xFFFF_FFFF


def _bits_d(f: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", f))[0] & 0xFFFF_FFFF_FFFF_FFFF


class TestFloatArithmetic:
    """Basic float arithmetic via handler."""

    def test_fadd_s_via_handler(self, eumos, state):
        """fadd.s: 1.0 + 2.0 = 3.0 in FPR."""
        s = state
        s.set_fpr(10, _bits_s(1.0))
        s.set_fpr(11, _bits_s(2.0))
        changes = float_ins.execute_fadd_s(
            {"rd": 10, "rs1": 11, "rs2": 10, "rm": 0},
            s,
            pc=0,
        )
        assert changes.fpr_writes
        assert changes.fpr_writes[0].register == 10
        assert changes.fpr_writes[0].value == _bits_s(3.0)
        assert s.get_fpr(10) == _bits_s(3.0)

    def test_fadd_d_via_handler(self, eumos, state):
        """fadd.d: 1.5 + 2.5 = 4.0 in FPR."""
        s = state
        s.set_fpr(1, _bits_d(1.5))
        s.set_fpr(2, _bits_d(2.5))
        changes = float_ins.execute_fadd_d(
            {"rd": 1, "rs1": 1, "rs2": 2, "rm": 0},
            s,
            pc=0,
        )
        assert changes.fpr_writes[0].register == 1
        assert changes.fpr_writes[0].value == _bits_d(4.0)


class TestFloatMove:
    """FMV instructions: GPR <-> FPR."""

    def test_fmv_w_x(self, state):
        """fmv.w.x: copy GPR low 32 bits to FPR."""
        s = state
        s.set_gpr(3, 0x40490FDB)
        changes = float_ins.execute_fmv_w_x({"rd": 5, "rs1": 3}, s, pc=0)
        assert s.get_fpr(5) == 0x40490FDB
        assert changes.fpr_writes[0].value == 0x40490FDB

    def test_fmv_x_w(self, state):
        """fmv.x.w: copy FPR low 32 bits to GPR (sign-extended)."""
        s = state
        s.set_fpr(5, 0x40490FDB)
        float_ins.execute_fmv_x_w({"rd": 3, "rs1": 5}, s, pc=0)
        assert s.get_gpr(3) == 0x40490FDB  # positive, no sign extend
        s.set_fpr(5, 0xBF800000)  # -1.0f
        float_ins.execute_fmv_x_w({"rd": 3, "rs1": 5}, s, pc=0)
        assert s.get_gpr(3) == 0xFFFF_FFFF_BF80_0000  # sign-extended


class TestFloatLoadStore:
    """FLW/FSW/FLD/FSD with mock memory."""

    def test_flw_fsw_round_trip(self, eumos):
        """FLW then FSW round-trip via memory."""
        from riscv_model.memory import MemoryInterface

        class SimpleMem(MemoryInterface):
            def __init__(self):
                self._data = {}

            def load(self, addr: int, size: int) -> int:
                v = 0
                for i in range(size):
                    v |= self._data.get(addr + i, 0) << (i * 8)
                return v

            def store(self, addr: int, value: int, size: int) -> None:
                for i in range(size):
                    self._data[addr + i] = (value >> (i * 8)) & 0xFF

        mem = SimpleMem()
        model = RISCVModel(eumos, memory=mem)
        state = model._state
        state.set_gpr(1, 0x1000)
        state.set_fpr(10, _bits_s(1.0))
        float_ins.execute_fsw(
            {"rs1": 1, "rs2": 10, "imm": 0},
            state,
            pc=0,
            memory=mem,
        )
        val = mem.load(0x1000, 4)
        assert val == _bits_s(1.0)
        state.set_fpr(10, 0)
        float_ins.execute_flw(
            {"rd": 10, "rs1": 1, "imm": 0},
            state,
            pc=0,
            memory=mem,
        )
        assert state.get_fpr(10) == _bits_s(1.0)


class TestModelFPRAccess:
    """Model exposes get_fpr/set_fpr/peek_fpr/poke_fpr."""

    def test_model_get_set_fpr(self, model):
        m = model
        m.set_fpr(1, 0x40490FDB)
        assert m.get_fpr(1) == 0x40490FDB

    def test_model_fpr_defs(self, model):
        m = model
        defs = m.fpr_defs
        if defs:
            assert 0 in defs
            assert defs[0].abi_name


class TestFCVTEdgeCases:
    """FCVT float-to-int: NaN, infinity, and overflow saturation."""

    def test_fcvt_w_s_inf_saturation(self, state):
        """fcvt.w.s: +inf -> 0x7FFF_FFFF, -inf -> 0x8000_0000 (sign-extended to XLEN)."""
        s = state
        s.set_fpr(1, _bits_s(float("inf")))
        float_ins.execute_fcvt_w_s({"rd": 2, "rs1": 1, "rm": 0}, s, pc=0)
        assert s.get_gpr(2) == 0x7FFF_FFFF
        s.set_fpr(1, _bits_s(float("-inf")))
        float_ins.execute_fcvt_w_s({"rd": 2, "rs1": 1, "rm": 0}, s, pc=0)
        # Result is 32-bit sign-extended to 64-bit: 0x8000_0000 -> 0xFFFF_FFFF_8000_0000
        assert (s.get_gpr(2) & 0xFFFF_FFFF) == 0x8000_0000

    def test_fcvt_wu_s_inf_saturation(self, state):
        """fcvt.wu.s: +inf -> 0xFFFF_FFFF; negative/NaN -> 0."""
        s = state
        s.set_fpr(1, _bits_s(float("inf")))
        float_ins.execute_fcvt_wu_s({"rd": 2, "rs1": 1, "rm": 0}, s, pc=0)
        assert s.get_gpr(2) == 0xFFFF_FFFF
        s.set_fpr(1, _bits_s(float("-inf")))
        float_ins.execute_fcvt_wu_s({"rd": 2, "rs1": 1, "rm": 0}, s, pc=0)
        assert s.get_gpr(2) == 0
        s.set_fpr(1, _bits_s(float("nan")))
        float_ins.execute_fcvt_wu_s({"rd": 2, "rs1": 1, "rm": 0}, s, pc=0)
        assert s.get_gpr(2) == 0

    def test_fcvt_w_s_finite_overflow_saturation(self, state):
        """fcvt.w.s: finite value beyond 32-bit signed range saturates."""
        s = state
        s.set_fpr(1, _bits_s(1e20))
        float_ins.execute_fcvt_w_s({"rd": 2, "rs1": 1, "rm": 0}, s, pc=0)
        assert s.get_gpr(2) == 0x7FFF_FFFF
        s.set_fpr(1, _bits_s(-1e20))
        float_ins.execute_fcvt_w_s({"rd": 2, "rs1": 1, "rm": 0}, s, pc=0)
        # Saturated to min signed 32-bit, sign-extended to 64-bit
        assert (s.get_gpr(2) & 0xFFFF_FFFF) == 0x8000_0000


class TestFCLASS:
    """FCLASS: classification mask bits per RISC-V (bits 0–9)."""

    def test_fclass_s_inf_zero_normal(self, state):
        """fclass.s: +inf (bit 7), -inf (bit 0), +0 (bit 4), -0 (bit 3), normal (bit 1/6)."""
        s = state
        s.set_fpr(1, _bits_s(float("inf")))
        float_ins.execute_fclass_s({"rd": 2, "rs1": 1}, s, pc=0)
        assert s.get_gpr(2) == (1 << 7)
        s.set_fpr(1, _bits_s(float("-inf")))
        float_ins.execute_fclass_s({"rd": 2, "rs1": 1}, s, pc=0)
        assert s.get_gpr(2) == (1 << 0)
        s.set_fpr(1, _bits_s(0.0))
        float_ins.execute_fclass_s({"rd": 2, "rs1": 1}, s, pc=0)
        assert s.get_gpr(2) == (1 << 4)
        s.set_fpr(1, _bits_s(-0.0))
        float_ins.execute_fclass_s({"rd": 2, "rs1": 1}, s, pc=0)
        assert s.get_gpr(2) == (1 << 3)
        s.set_fpr(1, _bits_s(1.5))
        float_ins.execute_fclass_s({"rd": 2, "rs1": 1}, s, pc=0)
        assert s.get_gpr(2) == (1 << 6)  # positive normal
        s.set_fpr(1, _bits_s(-1.5))
        float_ins.execute_fclass_s({"rd": 2, "rs1": 1}, s, pc=0)
        assert s.get_gpr(2) == (1 << 1)  # negative normal

    def test_fclass_s_subnormal(self, state):
        """fclass.s: positive subnormal -> bit 5, negative subnormal -> bit 2."""
        # Single-precision min positive subnormal
        tiny = 2.0 ** (-126 - 23)  # 2**-149
        s = state
        s.set_fpr(1, _bits_s(tiny))
        float_ins.execute_fclass_s({"rd": 2, "rs1": 1}, s, pc=0)
        assert s.get_gpr(2) == (1 << 5)
        s.set_fpr(1, _bits_s(-tiny))
        float_ins.execute_fclass_s({"rd": 2, "rs1": 1}, s, pc=0)
        assert s.get_gpr(2) == (1 << 2)


class TestFMinMaxNaN:
    """FMIN/FMAX with NaN operands: return the non-NaN or canonical NaN."""

    def test_fmin_s_one_nan(self, state):
        """fmin.s: if one operand is NaN, return the other."""
        s = state
        s.set_fpr(1, _bits_s(1.0))
        s.set_fpr(2, _bits_s(float("nan")))
        float_ins.execute_fmin_s({"rd": 3, "rs1": 1, "rs2": 2}, s, pc=0)
        assert s.get_fpr(3) == _bits_s(1.0)
        s.set_fpr(1, _bits_s(float("nan")))
        s.set_fpr(2, _bits_s(2.0))
        float_ins.execute_fmin_s({"rd": 3, "rs1": 1, "rs2": 2}, s, pc=0)
        assert s.get_fpr(3) == _bits_s(2.0)

    def test_fmax_s_one_nan(self, state):
        """fmax.s: if one operand is NaN, return the other."""
        s = state
        s.set_fpr(1, _bits_s(3.0))
        s.set_fpr(2, _bits_s(float("nan")))
        float_ins.execute_fmax_s({"rd": 3, "rs1": 1, "rs2": 2}, s, pc=0)
        assert s.get_fpr(3) == _bits_s(3.0)
