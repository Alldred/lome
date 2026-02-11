# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Tests for peek/poke and JSON export/restore on the RISCVModel level."""

import json

from riscv_model import RISCVModel

# ====================================================================
# GPR peek / poke via model
# ====================================================================


class TestModelGPRPeekPoke:
    """peek_gpr / poke_gpr through the model API."""

    def test_poke_and_peek(self, model):
        m = model
        old = m.poke_gpr(1, 42)
        assert old == 0
        assert m.peek_gpr(1) == 42

    def test_poke_gpr_then_get_gpr(self, model):
        m = model
        m.poke_gpr(1, 100)
        assert m.get_gpr(1) == 100

    def test_set_gpr_then_peek_gpr(self, model):
        m = model
        m.set_gpr(1, 50)
        assert m.peek_gpr(1) == 50

    def test_poke_x0(self, model):
        """poke_gpr allows writing x0 (raw)."""
        m = model
        m.poke_gpr(0, 123)
        assert m.peek_gpr(0) == 123
        assert m.get_gpr(0) == 0  # architectural still zero


# ====================================================================
# CSR peek / poke via model
# ====================================================================


class TestModelCSRPeekPoke:
    """peek_csr / poke_csr through the model API."""

    def test_poke_and_peek_by_address(self, model):
        m = model
        old = m.poke_csr(0x300, 0xABCD)
        assert old == 0
        assert m.peek_csr(0x300) == 0xABCD

    def test_poke_and_peek_by_name(self, model):
        m = model
        old = m.poke_csr("mstatus", 0x1234)
        assert old == 0
        assert m.peek_csr("mstatus") == 0x1234

    def test_set_csr_then_peek(self, model):
        m = model
        m.set_csr("mstatus", 0x42)
        assert m.peek_csr("mstatus") == 0x42

    def test_poke_nonexistent(self, model):
        m = model
        assert m.poke_csr(0xFFF, 1) is None
        assert m.peek_csr(0xFFF) is None

    def test_poke_nonexistent_name(self, model):
        m = model
        assert m.poke_csr("fake_csr", 1) is None
        assert m.peek_csr("fake_csr") is None


# ====================================================================
# PC peek / poke via model
# ====================================================================


class TestModelPCPeekPoke:
    """peek_pc / poke_pc through the model API."""

    def test_poke_and_peek_pc(self, model):
        m = model
        old = m.poke_pc(0x8000)
        assert old == 0
        assert m.peek_pc() == 0x8000

    def test_set_pc_then_peek(self, model):
        m = model
        m.set_pc(0x1000)
        assert m.peek_pc() == 0x1000

    def test_set_pc_returns_old(self, model):
        m = model
        m.set_pc(0x100)
        old = m.set_pc(0x200)
        assert old == 0x100


# ====================================================================
# JSON export / restore via model
# ====================================================================


class TestModelJSONExportRestore:
    """export_state / restore_state / export_state_json / from_json on model."""

    def test_export_state_dict(self, model):
        m = model
        m.poke_gpr(1, 42)
        m.poke_pc(0x1000)
        data = m.export_state()
        assert data["gprs"]["1"] == 42
        assert data["pc"] == 0x1000

    def test_restore_state(self, model, decoder, gpr_defs, csr_defs):
        m = model
        m.poke_gpr(1, 42)
        m.poke_pc(0x1000)
        data = m.export_state()

        m2 = RISCVModel(decoder=decoder, gpr_defs=gpr_defs, csr_defs=csr_defs)
        m2.restore_state(data)
        assert m2.get_gpr(1) == 42
        assert m2.get_pc() == 0x1000

    def test_restore_clears_last_changes(self, model):
        m = model
        addi = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (5 << 20)
        m.execute(addi)
        assert m.get_changes() is not None
        m.restore_state(m.export_state())
        assert m.get_changes() is None

    def test_export_state_json(self, model):
        m = model
        m.poke_gpr(3, 77)
        json_str = m.export_state_json()
        data = json.loads(json_str)
        assert data["gprs"]["3"] == 77

    def test_from_json(self, model, decoder, gpr_defs, csr_defs):
        m = model
        m.poke_gpr(5, 123)
        m.poke_pc(0x2000)
        m2 = RISCVModel.from_json(
            m.export_state_json(),
            decoder=decoder,
            gpr_defs=gpr_defs,
            csr_defs=csr_defs,
        )
        assert m2.get_gpr(5) == 123
        assert m2.get_pc() == 0x2000

    def test_full_round_trip(self, model, decoder, gpr_defs, csr_defs):
        """export -> JSON string -> from_json -> export -> compare."""
        m = model
        for i in range(1, 32):
            m.poke_gpr(i, i * 1000)
        m.poke_pc(0x4000)
        m.poke_csr(0x300, 0xDEAD)

        json_str = m.export_state_json()
        m2 = RISCVModel.from_json(
            json_str,
            decoder=decoder,
            gpr_defs=gpr_defs,
            csr_defs=csr_defs,
        )

        for i in range(1, 32):
            assert m2.get_gpr(i) == i * 1000
        assert m2.get_pc() == 0x4000
        assert m2.get_csr(0x300) == 0xDEAD

    def test_export_is_json_serialisable(self, model):
        """All values in export_state() must be JSON-serialisable."""
        m = model
        m.poke_gpr(1, 0xFFFF_FFFF_FFFF_FFFF)
        data = m.export_state()
        json_str = json.dumps(data)
        assert json_str  # non-empty

    def test_export_import_preserves_csr_values(
        self, model, decoder, gpr_defs, csr_defs
    ):
        m = model
        m.poke_csr(0x300, 0xCAFE)
        data = m.export_state()

        m2 = RISCVModel(decoder=decoder, gpr_defs=gpr_defs, csr_defs=csr_defs)
        m2.restore_state(data)
        assert m2.get_csr(0x300) == 0xCAFE
