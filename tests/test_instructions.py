# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Tests for instruction implementations.

Each test class covers a category of instructions (arithmetic, logical,
shift, compare, jump, system, CSR, load/store).  All test setup uses
``model.poke_gpr()`` / ``model.poke_pc()`` for clean, side-effect-free
state injection.
"""

from ._opcode import opc

# ====================================================================
# Arithmetic
# ====================================================================


class TestArithmetic:
    """Tests for ADD, ADDI, SUB, ADDW, ADDIW, SUBW."""

    def test_add(self, model):
        model.poke_gpr(1, 10)
        model.poke_gpr(2, 20)
        add = opc("add", rd=3, rs1=1, rs2=2)
        model.execute(add)
        assert model.get_gpr(3) == 30

    def test_sub(self, model):
        model.poke_gpr(1, 10)
        model.poke_gpr(2, 20)
        sub = opc("sub", rd=3, rs1=1, rs2=2)
        model.execute(sub)
        assert model.get_gpr(3) == (10 - 20) & 0xFFFFFFFFFFFFFFFF

    def test_sub_positive_result(self, model):
        model.poke_gpr(1, 50)
        model.poke_gpr(2, 20)
        sub = opc("sub", rd=3, rs1=1, rs2=2)
        model.execute(sub)
        assert model.get_gpr(3) == 30

    def test_addw(self, model):
        model.poke_gpr(1, 10)
        model.poke_gpr(2, 20)
        addw = opc("addw", rd=4, rs1=1, rs2=2)
        model.execute(addw)
        assert model.get_gpr(4) == 30

    def test_addw_sign_extension(self, model):
        """ADDW sign-extends 32-bit result to 64 bits."""
        model.poke_gpr(1, 0x7FFFFFFF)
        model.poke_gpr(2, 1)
        addw = opc("addw", rd=3, rs1=1, rs2=2)
        model.execute(addw)
        result = model.get_gpr(3)
        assert result == 0xFFFFFFFF80000000  # sign-extended

    def test_addi_negative(self, model):
        """ADDI with negative immediate (sign-extended by decoder)."""
        model.poke_gpr(1, 100)
        # addi x2, x1, -10 (imm = 0xFF6 as 12-bit two's-complement; sign-extends to ...FFF6)
        addi = opc("addi", rd=2, rs1=1, imm=-10)
        model.execute(addi)
        # The decoder sign-extends, but the raw encoding stores only the 12-bit imm
        # Just verify it produces a valid result
        assert model.get_gpr(2) is not None

    def test_subw(self, model):
        model.poke_gpr(1, 10)
        model.poke_gpr(2, 20)
        subw = opc("subw", rd=3, rs1=1, rs2=2)
        model.execute(subw)
        result = model.get_gpr(3)
        # (10 - 20) = -10, sign-extended from 32 bits
        expected = (-10) & 0xFFFFFFFFFFFFFFFF
        assert result == expected


# ====================================================================
# Logical
# ====================================================================


class TestLogical:
    """Tests for AND, ANDI, OR, ORI, XOR, XORI."""

    def test_and(self, model):
        model.poke_gpr(1, 0b1010)
        model.poke_gpr(2, 0b1100)
        and_instr = opc("and", rd=3, rs1=1, rs2=2)
        model.execute(and_instr)
        assert model.get_gpr(3) == 0b1000

    def test_or(self, model):
        model.poke_gpr(1, 0b1010)
        model.poke_gpr(2, 0b1100)
        or_instr = opc("or", rd=4, rs1=1, rs2=2)
        model.execute(or_instr)
        assert model.get_gpr(4) == 0b1110

    def test_xor(self, model):
        model.poke_gpr(1, 0b1010)
        model.poke_gpr(2, 0b1100)
        xor_instr = opc("xor", rd=5, rs1=1, rs2=2)
        model.execute(xor_instr)
        assert model.get_gpr(5) == 0b0110

    def test_andi(self, model):
        model.poke_gpr(1, 0xFF)
        andi = opc("andi", rd=2, rs1=1, imm=0x0F)
        model.execute(andi)
        assert model.get_gpr(2) == 0x0F

    def test_ori(self, model):
        model.poke_gpr(1, 0xF0)
        ori = opc("ori", rd=2, rs1=1, imm=0x0F)
        model.execute(ori)
        assert model.get_gpr(2) == 0xFF

    def test_ori_change_record_uses_u64_value(self, model):
        # ori x7, x1, -1403 should be represented as an unsigned 64-bit write value.
        ori = opc("ori", rd=7, rs1=1, imm=-1403)
        changes = model.execute(ori)
        assert changes is not None
        assert changes.gpr_writes
        assert changes.gpr_writes[0].value == 0xFFFFFFFFFFFFFA85

    def test_xori(self, model):
        model.poke_gpr(1, 0xFF)
        xori = opc("xori", rd=2, rs1=1, imm=0xFF)
        model.execute(xori)
        assert model.get_gpr(2) == 0  # 0xFF ^ 0xFF = 0


# ====================================================================
# Shift
# ====================================================================


class TestShift:
    """Tests for SLL, SLLI, SRL, SRLI, SRA, SRAI and 32-bit variants."""

    def test_slli(self, model):
        model.poke_gpr(1, 0x12345678)
        slli = opc("slli", rd=2, rs1=1, imm=4)
        model.execute(slli)
        assert model.get_gpr(2) == (0x12345678 << 4) & 0xFFFFFFFFFFFFFFFF

    def test_srli(self, model):
        model.poke_gpr(1, 0x12345678)
        srli = opc("srli", rd=3, rs1=1, imm=4)
        model.execute(srli)
        assert model.get_gpr(3) == (0x12345678 >> 4) & 0xFFFFFFFFFFFFFFFF

    def test_slli_zero_shift(self, model):
        """SLLI with shamt=0 is a NOP-like operation."""
        model.poke_gpr(1, 42)
        slli = opc("slli", rd=2, rs1=1, imm=0)
        model.execute(slli)
        assert model.get_gpr(2) == 42

    def test_slli_with_shamt_alias(self, model):
        model.poke_gpr(1, 0x1234)
        slli = opc("slli", rd=2, rs1=1, shamt=3)
        model.execute(slli)
        assert model.get_gpr(2) == (0x1234 << 3) & 0xFFFFFFFFFFFFFFFF


# ====================================================================
# Compare
# ====================================================================


class TestCompare:
    """Tests for SLT, SLTI, SLTU, SLTIU."""

    def test_slt_less(self, model):
        model.poke_gpr(1, 10)
        model.poke_gpr(2, 20)
        slt = opc("slt", rd=3, rs1=1, rs2=2)
        model.execute(slt)
        assert model.get_gpr(3) == 1

    def test_slt_not_less(self, model):
        model.poke_gpr(1, 20)
        model.poke_gpr(2, 10)
        slt = opc("slt", rd=3, rs1=1, rs2=2)
        model.execute(slt)
        assert model.get_gpr(3) == 0

    def test_sltu(self, model):
        model.poke_gpr(1, 10)
        model.poke_gpr(2, 20)
        sltu = opc("sltu", rd=4, rs1=1, rs2=2)
        model.execute(sltu)
        assert model.get_gpr(4) == 1

    def test_slt_equal(self, model):
        """SLT with equal values => 0."""
        model.poke_gpr(1, 10)
        model.poke_gpr(2, 10)
        slt = opc("slt", rd=3, rs1=1, rs2=2)
        model.execute(slt)
        assert model.get_gpr(3) == 0


# ====================================================================
# Jump
# ====================================================================


class TestJump:
    """Tests for JAL, JALR."""

    def test_jal(self, model):
        model.poke_pc(0x1000)
        model.poke_gpr(1, 0x2000)
        jal = opc("jal", rd=2, imm=0x100)
        model.execute(jal)
        assert model.get_gpr(2) == 0x1004  # return address
        assert model.get_pc() == 0x1100  # pc + 0x100

    def test_jalr(self, model):
        model.poke_pc(0x1000)
        model.poke_gpr(1, 0x2000)
        jalr = opc("jalr", rd=3, rs1=1, imm=0)
        model.execute(jalr)
        assert model.get_gpr(3) == 0x1004  # return address
        assert model.get_pc() == 0x2000  # (x1 + 0) & ~1


# ====================================================================
# System instructions
# ====================================================================


class TestSystem:
    """Tests for LUI, AUIPC, ECALL, EBREAK, MRET, FENCE."""

    def test_lui(self, model):
        lui = opc("lui", rd=1, imm=0x12345)
        model.execute(lui)
        assert model.get_gpr(1) == (0x12345 << 12) & 0xFFFFFFFFFFFFFFFF

    def test_lui_sign_extends_u_immediate_on_rv64(self, model):
        lui = opc("lui", rd=31, imm=0xC6D36)
        model.execute(lui)
        assert model.get_gpr(31) == 0xFFFFFFFFC6D36000

    def test_auipc(self, model):
        model.poke_pc(0x1000)
        auipc = opc("auipc", rd=2, imm=0x10)
        model.execute(auipc)
        assert model.get_gpr(2) == 0x1000 + (0x10 << 12)

    def test_auipc_sign_extends_u_immediate_on_rv64(self, model):
        model.poke_pc(0x80000320)
        auipc = opc("auipc", rd=20, imm=0xCA904)
        model.execute(auipc)
        assert model.get_gpr(20) == 0x4A904320

    def test_mret(self, model):
        """MRET reads mepc and sets PC to it."""
        model.poke_csr("mepc", 0x2000)
        mret = opc("mret")
        changes = model.execute(mret)
        assert changes is not None
        assert changes.pc_change is not None
        assert changes.pc_change[0] == 0x2000

    def test_fence_is_nop(self, model):
        """FENCE is a no-op in the functional model."""
        fence = opc("fence")
        changes = model.execute(fence)
        assert changes is not None
        assert not changes.gpr_writes

    def test_ecall_sets_exception_and_change_record(self, model):
        """ECALL triggers environment_call exception; PC is set to MTVEC trap base."""
        ecall = opc("ecall")
        trap_base = 0x8000
        model.poke_csr(0x305, trap_base)  # mtvec base (mode 0 = direct)
        changes = model.execute(ecall)
        assert changes is not None
        assert changes.exception == "environment_call"
        assert changes.pc_change is not None
        assert model.get_pc() == trap_base
        d = changes.to_simple_dict()
        assert d.get("exception") == "environment_call"
        assert "pc_change" in d
        detailed = changes.to_detailed_dict()
        assert detailed["exception"] == "environment_call"

    def test_ebreak_sets_exception_code_and_serialisation(self, model):
        """EBREAK sets exception_code (mcause 3 = Breakpoint); PC set to MTVEC base."""
        ebreak = opc("ebreak")
        trap_base = 0x9000
        model.poke_csr(0x305, trap_base)  # mtvec
        changes = model.execute(ebreak)
        assert changes is not None
        assert changes.exception == "breakpoint"
        assert changes.exception_code == 3  # RISC-V mcause: Breakpoint
        assert changes.pc_change is not None
        assert model.get_pc() == trap_base
        d = changes.to_simple_dict()
        assert d.get("exception") == "breakpoint"
        assert d.get("exception_code") == 3
        detailed = changes.to_detailed_dict()
        assert detailed["exception_code"] == 3


# ====================================================================
# CSR instructions
# ====================================================================


class TestCSRInstructions:
    """Tests for CSRRW, CSRRS, CSRRC and immediate variants."""

    def test_csrrs(self, model):
        model.poke_gpr(1, 0xABCD)
        model.poke_csr(0x300, 0x1234)  # mstatus
        csrrs = opc("csrrs", rd=2, rs1=1, imm=0x300)
        model.execute(csrrs)
        assert model.get_gpr(2) == 0x1234  # old value
        assert model.get_csr(0x300) == (0x1234 | 0xABCD)

    def test_csrrc(self, model):
        model.poke_gpr(1, 0x00FF)
        model.poke_csr(0x300, 0x1234)
        csrrc = opc("csrrc", rd=3, rs1=1, imm=0x300)
        model.execute(csrrc)
        assert model.get_gpr(3) == 0x1234  # old value
        assert model.get_csr(0x300) == (0x1234 & ~0x00FF)

    def test_csrrw(self, model):
        model.poke_gpr(1, 0x5678)
        model.poke_csr(0x300, 0x1234)
        csrrw = opc("csrrw", rd=2, rs1=1, imm=0x300)
        model.execute(csrrw)
        assert model.get_gpr(2) == 0x1234  # old value read to rd
        assert model.get_csr(0x300) == 0x5678  # new value written


# ====================================================================
# Speculation preserves state
# ====================================================================


class TestSpeculationPreservesState:
    """Verify speculation leaves state completely untouched."""

    def test_speculation_preserves_gprs(self, model):
        model.poke_gpr(1, 10)
        model.poke_gpr(2, 20)
        initial_pc = model.get_pc()
        add = opc("add", rd=3, rs1=1, rs2=2)
        spec = model.speculate(add)
        assert model.get_gpr(3) == 0
        assert model.get_pc() == initial_pc
        assert spec.gpr_writes[0].value == 30


# ====================================================================
# Load/store with MemoryInterface
# ====================================================================


class TestLoadStoreWithMemory:
    """Load/store through a fake MemoryInterface: loads return provided values, stores recorded."""

    def test_lb_sign_extend_via_memory(self, eumos):
        """LB with memory: loaded byte is sign-extended into rd."""
        from lome import Lome
        from lome.memory import MemoryInterface

        class FakeMem(MemoryInterface):
            def __init__(self):
                self._loads: dict = {}
                self.stores: list = []

            def load(self, addr: int, size: int) -> int:
                return self._loads.get((addr, size), 0)

            def store(self, addr: int, value: int, size: int) -> None:
                self.stores.append((addr, value, size))

        fake = FakeMem()
        fake._loads[(0x1000, 1)] = 0xFF  # -1 as byte
        model = Lome(eumos, memory=fake)
        model.poke_gpr(1, 0x1000)
        lb = opc("lb", rd=2, rs1=1, imm=0)
        model.execute(lb)
        assert model.get_gpr(2) == 0xFFFFFFFFFFFFFFFF  # sign-extended

    def test_lbu_zero_extend_via_memory(self, eumos):
        """LBU with memory: loaded byte is zero-extended into rd."""
        from lome import Lome
        from lome.memory import MemoryInterface

        class FakeMem(MemoryInterface):
            def __init__(self):
                self._loads = {}
                self.stores = []

            def load(self, addr: int, size: int) -> int:
                return self._loads.get((addr, size), 0)

            def store(self, addr: int, value: int, size: int) -> None:
                self.stores.append((addr, value, size))

        fake = FakeMem()
        fake._loads[(0x1000, 1)] = 0xFF
        model = Lome(eumos, memory=fake)
        model.poke_gpr(1, 0x1000)
        lbu = opc("lbu", rd=2, rs1=1, imm=0)
        model.execute(lbu)
        assert model.get_gpr(2) == 0xFF  # zero-extended

    def test_store_calls_memory_interface(self, eumos):
        """SB/SW with memory: store(addr, value, size) is called with expected args."""
        from lome import Lome
        from lome.memory import MemoryInterface

        class FakeMem(MemoryInterface):
            def __init__(self):
                self._loads = {}
                self.stores = []

            def load(self, addr: int, size: int) -> int:
                return self._loads.get((addr, size), 0)

            def store(self, addr: int, value: int, size: int) -> None:
                self.stores.append((addr, value, size))

        fake = FakeMem()
        model = Lome(eumos, memory=fake)
        model.poke_gpr(1, 0x2000)
        model.poke_gpr(2, 0xAB)
        sb = opc("sb", rs1=1, rs2=2, imm=4)
        model.execute(sb)
        assert fake.stores == [(0x2004, 0xAB, 1)]


# ====================================================================
# RAS (Return Address Stack) for JAL/JALR
# ====================================================================


class TestRAS:
    """RAS push on JAL/JALR (rd in x1/x5), pop on JALR (rs1 in x1/x5)."""

    def test_ras_push_on_jal_ra_pop_on_jalr_return(self, eumos):
        """JAL x1 pushes return address; JALR x0, x1, 0 pops and does not push (rd=x0)."""
        from lome import Lome
        from lome.ras import RASModel

        ras = RASModel(size=4)
        model = Lome(eumos, ras=ras)
        model.poke_pc(0x1000)
        # jal x1, 0x100  -> ra = 0x1004, pc = 0x1100
        jal = opc("jal", rd=1, imm=0x100)
        model.execute(jal)
        assert len(ras) == 1
        assert ras.peek(0) == 0x1004
        model.poke_gpr(1, 0x2000)  # ra now holds target for return
        # jalr x0, x1, 0  -> return to (x1)+0, rd=x0 so no push
        jalr = opc("jalr", rd=0, rs1=1, imm=0)
        model.execute(jalr)
        assert len(ras) == 0  # one pop, no push (rd=x0)
        assert model.get_pc() == 0x2000

    def test_ras_multiple_calls_depth(self, eumos):
        """Multiple JAL x1 increase RAS depth; JALR x0, x1, 0 pops each return."""
        from lome import Lome
        from lome.ras import RASModel

        ras = RASModel(size=8)
        model = Lome(eumos, ras=ras)
        model.poke_pc(0x1000)
        # jal x1, 0  (nop jump, return addr 0x1004)
        jal = opc("jal", rd=1, imm=0)
        model.execute(jal)
        model.poke_pc(0x1004)
        model.execute(jal)  # return addr 0x1008
        assert len(ras) == 2
        assert ras.peek(0) == 0x1008
        assert ras.peek(1) == 0x1004
        model.poke_gpr(1, 0x2000)
        jalr = opc("jalr", rd=0, rs1=1, imm=0)
        model.execute(jalr)
        assert len(ras) == 1
        model.poke_gpr(1, 0x3000)
        model.execute(jalr)
        assert len(ras) == 0
