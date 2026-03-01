# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Additional instruction tests for coverage of branch, shift, compare,
load/store, and system instruction variants.

These tests encode instructions via Eumos and verify outcomes through
the model's public API.
"""

from ._opcode import opc

# ====================================================================
# Branch instructions -- BNE, BLT, BGE, BLTU, BGEU
# ====================================================================


class TestBranchVariants:
    """Coverage for branch instructions beyond BEQ."""

    def test_bne_taken(self, model):
        m = model
        m.poke_gpr(1, 10)
        m.poke_gpr(2, 20)
        bne = opc("bne", rs1=1, rs2=2, imm=16)  # funct3=1 = BNE, offset 16
        changes = m.execute(bne)
        assert changes.branch_info.taken is True
        assert changes.branch_info.condition == "ne"

    def test_bne_not_taken(self, model):
        m = model
        m.poke_gpr(1, 10)
        m.poke_gpr(2, 10)
        bne = opc("bne", rs1=1, rs2=2, imm=16)
        changes = m.execute(bne)
        assert changes.branch_info.taken is False

    def test_blt_taken(self, model):
        m = model
        m.poke_gpr(1, 5)
        m.poke_gpr(2, 10)
        blt = opc("blt", rs1=1, rs2=2, imm=8)  # funct3=4 = BLT
        changes = m.execute(blt)
        assert changes.branch_info.taken is True
        assert changes.branch_info.condition == "lt"

    def test_blt_not_taken(self, model):
        m = model
        m.poke_gpr(1, 20)
        m.poke_gpr(2, 10)
        blt = opc("blt", rs1=1, rs2=2, imm=8)
        changes = m.execute(blt)
        assert changes.branch_info.taken is False

    def test_bge_taken(self, model):
        m = model
        m.poke_gpr(1, 10)
        m.poke_gpr(2, 10)
        bge = opc("bge", rs1=1, rs2=2, imm=8)  # funct3=5 = BGE
        changes = m.execute(bge)
        assert changes.branch_info.taken is True
        assert changes.branch_info.condition == "ge"

    def test_bge_not_taken(self, model):
        m = model
        m.poke_gpr(1, 5)
        m.poke_gpr(2, 10)
        bge = opc("bge", rs1=1, rs2=2, imm=8)
        changes = m.execute(bge)
        assert changes.branch_info.taken is False

    def test_bltu_taken(self, model):
        m = model
        m.poke_gpr(1, 5)
        m.poke_gpr(2, 10)
        bltu = opc("bltu", rs1=1, rs2=2, imm=8)  # funct3=6 = BLTU
        changes = m.execute(bltu)
        assert changes.branch_info.taken is True
        assert changes.branch_info.condition == "ltu"

    def test_bltu_not_taken(self, model):
        m = model
        m.poke_gpr(1, 20)
        m.poke_gpr(2, 10)
        bltu = opc("bltu", rs1=1, rs2=2, imm=8)
        changes = m.execute(bltu)
        assert changes.branch_info.taken is False

    def test_bgeu_taken(self, model):
        m = model
        m.poke_gpr(1, 10)
        m.poke_gpr(2, 10)
        bgeu = opc("bgeu", rs1=1, rs2=2, imm=8)  # funct3=7 = BGEU
        changes = m.execute(bgeu)
        assert changes.branch_info.taken is True
        assert changes.branch_info.condition == "geu"

    def test_bgeu_not_taken(self, model):
        m = model
        m.poke_gpr(1, 5)
        m.poke_gpr(2, 10)
        bgeu = opc("bgeu", rs1=1, rs2=2, imm=8)
        changes = m.execute(bgeu)
        assert changes.branch_info.taken is False


# ====================================================================
# Shift instructions -- SLL, SRL, SRA, SRAI, and word variants
# ====================================================================


class TestShiftVariants:
    """Coverage for shift instructions beyond SLLI/SRLI."""

    def test_sll(self, model):
        m = model
        m.poke_gpr(1, 0x1)
        m.poke_gpr(2, 4)
        sll = opc("sll", rd=3, rs1=1, rs2=2)  # SLL
        m.execute(sll)
        assert m.get_gpr(3) == 0x10

    def test_srl(self, model):
        m = model
        m.poke_gpr(1, 0x100)
        m.poke_gpr(2, 4)
        srl = opc("srl", rd=3, rs1=1, rs2=2)  # SRL
        m.execute(srl)
        assert m.get_gpr(3) == 0x10

    def test_sra(self, model):
        m = model
        # Negative number: SRA should sign-extend
        m.poke_gpr(1, 0xFFFFFFFFFFFFFF00)  # -256
        m.poke_gpr(2, 4)
        sra = opc("sra", rd=3, rs1=1, rs2=2)  # SRA (funct7=0x20)
        m.execute(sra)
        result = m.get_gpr(3)
        # -256 >> 4 = -16, which is 0xFFFFFFFFFFFFFFF0 in 64-bit two's complement
        assert result == 0xFFFFFFFFFFFFFFF0

    def test_srai(self, model):
        m = model
        m.poke_gpr(1, 0xFFFFFFFFFFFFFF00)  # -256
        # SRAI: I-type with funct7=0x20 in bits [31:25], shamt in bits [24:20]
        srai = opc("srai", rd=3, rs1=1, shamt=4)  # shamt=4, funct7=0x20
        m.execute(srai)
        result = m.get_gpr(3)
        assert result & 0x8000000000000000  # should still be negative

    def test_sllw(self, model):
        m = model
        m.poke_gpr(1, 0x1)
        m.poke_gpr(2, 4)
        sllw = opc("sllw", rd=3, rs1=1, rs2=2)  # SLLW
        m.execute(sllw)
        assert m.get_gpr(3) == 0x10

    def test_slliw(self, model):
        m = model
        m.poke_gpr(1, 0x7FFFFFFF)
        slliw = opc("slliw", rd=3, rs1=1, shamt=1)  # SLLIW, shamt=1
        m.execute(slliw)
        result = m.get_gpr(3)
        # 0x7FFFFFFF << 1 = 0xFFFFFFFE -> sign-extended to 64 bits
        assert result == 0xFFFFFFFFFFFFFFFE

    def test_srlw(self, model):
        m = model
        m.poke_gpr(1, 0x100)
        m.poke_gpr(2, 4)
        srlw = opc("srlw", rd=3, rs1=1, rs2=2)  # SRLW
        m.execute(srlw)
        assert m.get_gpr(3) == 0x10

    def test_srliw(self, model):
        m = model
        m.poke_gpr(1, 0x80000000)  # bit 31 set
        srliw = opc("srliw", rd=3, rs1=1, shamt=1)  # SRLIW, shamt=1
        m.execute(srliw)
        assert m.get_gpr(3) == 0x40000000

    def test_sraw(self, model):
        m = model
        m.poke_gpr(1, 0x80000000)  # bit 31 set (negative in 32-bit)
        m.poke_gpr(2, 4)
        sraw = opc("sraw", rd=3, rs1=1, rs2=2)  # SRAW
        m.execute(sraw)
        result = m.get_gpr(3)
        assert result & 0x8000000000000000  # should be sign-extended to negative

    def test_sraiw(self, model):
        m = model
        m.poke_gpr(1, 0x80000000)
        sraiw = opc("sraiw", rd=3, rs1=1, shamt=4)  # SRAIW, shamt=4
        m.execute(sraiw)
        result = m.get_gpr(3)
        assert result & 0x8000000000000000  # sign-extended


# ====================================================================
# Compare instructions -- SLTI, SLTIU
# ====================================================================


class TestCompareVariants:
    """Coverage for SLTI and SLTIU."""

    def test_slti_less(self, model):
        m = model
        m.poke_gpr(1, 5)
        slti = opc("slti", rd=3, rs1=1, imm=10)  # SLTI: rd=3, rs1=1, imm=10
        m.execute(slti)
        assert m.get_gpr(3) == 1

    def test_slti_not_less(self, model):
        m = model
        m.poke_gpr(1, 20)
        slti = opc("slti", rd=3, rs1=1, imm=10)
        m.execute(slti)
        assert m.get_gpr(3) == 0

    def test_sltiu_less(self, model):
        m = model
        m.poke_gpr(1, 5)
        sltiu = opc("sltiu", rd=3, rs1=1, imm=10)  # SLTIU: funct3=3
        m.execute(sltiu)
        assert m.get_gpr(3) == 1

    def test_sltiu_not_less(self, model):
        m = model
        m.poke_gpr(1, 20)
        sltiu = opc("sltiu", rd=3, rs1=1, imm=10)
        m.execute(sltiu)
        assert m.get_gpr(3) == 0


# ====================================================================
# Load/store instructions
# ====================================================================


class TestLoadStore:
    """Coverage for load/store instruction variants.

    Since the memory model is not yet implemented, loads return 0 and
    stores just track the access.  We verify the change records.
    """

    def test_lb(self, model):
        m = model
        m.poke_gpr(1, 0x1000)
        lb = opc("lb", rd=2, rs1=1, imm=4)  # LB: rd=2, rs1=1, imm=4
        changes = m.execute(lb)
        assert len(changes.memory_accesses) == 1
        assert changes.memory_accesses[0].address == 0x1004
        assert changes.memory_accesses[0].size == 1
        assert not changes.memory_accesses[0].is_write

    def test_lh(self, model):
        m = model
        m.poke_gpr(1, 0x1000)
        lh = opc("lh", rd=2, rs1=1, imm=0)  # LH
        changes = m.execute(lh)
        assert changes.memory_accesses[0].size == 2

    def test_lw(self, model):
        m = model
        m.poke_gpr(1, 0x1000)
        lw = opc("lw", rd=2, rs1=1, imm=0)  # LW
        changes = m.execute(lw)
        assert changes.memory_accesses[0].size == 4

    def test_ld(self, model):
        m = model
        m.poke_gpr(1, 0x1000)
        ld = opc("ld", rd=2, rs1=1, imm=0)  # LD
        changes = m.execute(ld)
        assert changes.memory_accesses[0].size == 8

    def test_lbu(self, model):
        m = model
        m.poke_gpr(1, 0x1000)
        lbu = opc("lbu", rd=2, rs1=1, imm=0)  # LBU
        changes = m.execute(lbu)
        assert changes.memory_accesses[0].size == 1

    def test_lhu(self, model):
        m = model
        m.poke_gpr(1, 0x1000)
        lhu = opc("lhu", rd=2, rs1=1, imm=0)  # LHU
        changes = m.execute(lhu)
        assert changes.memory_accesses[0].size == 2

    def test_lwu(self, model):
        m = model
        m.poke_gpr(1, 0x1000)
        lwu = opc("lwu", rd=2, rs1=1, imm=0)  # LWU
        changes = m.execute(lwu)
        assert changes.memory_accesses[0].size == 4

    def test_sb(self, model):
        m = model
        m.poke_gpr(1, 0x1000)
        m.poke_gpr(2, 0xFF)
        sb = opc("sb", rs1=1, rs2=2, imm=4)  # SB: rs1=1, rs2=2, imm=4
        changes = m.execute(sb)
        assert len(changes.memory_accesses) == 1
        assert changes.memory_accesses[0].address == 0x1004
        assert changes.memory_accesses[0].size == 1
        assert changes.memory_accesses[0].is_write
        assert changes.memory_accesses[0].value == 0xFF

    def test_sh(self, model):
        m = model
        m.poke_gpr(1, 0x1000)
        m.poke_gpr(2, 0xFFFF)
        sh = opc("sh", rs1=1, rs2=2, imm=0)  # SH
        changes = m.execute(sh)
        assert changes.memory_accesses[0].size == 2
        assert changes.memory_accesses[0].value == 0xFFFF

    def test_sw(self, model):
        m = model
        m.poke_gpr(1, 0x1000)
        m.poke_gpr(2, 0xDEADBEEF)
        sw = opc("sw", rs1=1, rs2=2, imm=0)  # SW
        changes = m.execute(sw)
        assert changes.memory_accesses[0].size == 4
        assert changes.memory_accesses[0].value == 0xDEADBEEF

    def test_sd(self, model):
        m = model
        m.poke_gpr(1, 0x1000)
        m.poke_gpr(2, 0x123456789ABCDEF0)
        sd = opc("sd", rs1=1, rs2=2, imm=0)  # SD
        changes = m.execute(sd)
        assert changes.memory_accesses[0].size == 8
        assert changes.memory_accesses[0].value == 0x123456789ABCDEF0


# ====================================================================
# System instructions -- CSR immediate variants
# ====================================================================


class TestCSRImmediateVariants:
    """Coverage for CSRRWI, CSRRSI, CSRRCI."""

    def test_csrrwi(self, model):
        """CSRRWI: rd = CSR; CSR = zimm."""
        m = model
        m.poke_csr(0x300, 0x1234)
        # csrrwi x2, mstatus, 5
        # funct3=5 for CSRRWI, rs1 field encodes zimm=5
        csrrwi = opc("csrrwi", rd=2, rs1=5, imm=0x300)
        m.execute(csrrwi)
        assert m.get_gpr(2) == 0x1234  # old CSR value
        assert m.get_csr(0x300) == 5  # zimm written

    def test_csrrsi(self, model):
        """CSRRSI: rd = CSR; CSR = CSR | zimm."""
        m = model
        m.poke_csr(0x300, 0x10)
        # csrrsi x2, mstatus, 3 (zimm=3)
        csrrsi = opc("csrrsi", rd=2, rs1=3, imm=0x300)
        m.execute(csrrsi)
        assert m.get_gpr(2) == 0x10
        assert m.get_csr(0x300) == (0x10 | 3)

    def test_csrrci(self, model):
        """CSRRCI: rd = CSR; CSR = CSR & ~zimm."""
        m = model
        m.poke_csr(0x300, 0xFF)
        # csrrci x2, mstatus, 0x0F (zimm=15)
        csrrci = opc("csrrci", rd=2, rs1=15, imm=0x300)
        m.execute(csrrci)
        assert m.get_gpr(2) == 0xFF
        assert m.get_csr(0x300) == (0xFF & ~15)


# ====================================================================
# Arithmetic word variants
# ====================================================================


class TestArithmeticWordVariants:
    """Coverage for ADDIW."""

    def test_addiw(self, model):
        m = model
        m.poke_gpr(1, 10)
        addiw = opc("addiw", rd=2, rs1=1, imm=5)  # ADDIW: rd=2, rs1=1, imm=5
        m.execute(addiw)
        assert m.get_gpr(2) == 15

    def test_addiw_sign_extension(self, model):
        m = model
        m.poke_gpr(1, 0x7FFFFFFF)
        addiw = opc("addiw", rd=2, rs1=1, imm=1)  # ADDIW: add 1 -> overflow 32-bit
        m.execute(addiw)
        result = m.get_gpr(2)
        assert result == 0xFFFFFFFF80000000  # sign-extended


# ====================================================================
# Executor edge cases
# ====================================================================


class TestExecutorEdgeCases:
    """Coverage for executor error paths."""

    def test_none_instruction_returns_none(self, state):
        """execute_instruction(None, ...) returns None."""
        from lome.executor import execute_instruction

        s = state
        assert execute_instruction(None, s, 0) is None

    def test_speculation_restores_state(self, model):
        """Speculation executes but does not commit changes to the model."""
        m = model
        m.poke_gpr(1, 42)
        # Execute a valid instruction in speculation mode
        addi = opc("addi", rd=2, rs1=1, imm=5)
        changes = m.speculate(addi)
        assert changes is not None
        assert changes.gpr_writes[0].value == 47  # 42 + 5
        assert m.get_gpr(2) == 0  # state unchanged
        assert m.get_gpr(1) == 42  # preserved
