# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred. All Rights Reserved

"""Example usage of RISC-V functional model."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from riscv_model import RISCVModel


def main():
    """Demonstrate basic operations, speculation, and change queries."""
    model = RISCVModel()

    print("=== RISC-V Functional Model Example ===\n")

    # Example 1: Simple arithmetic
    print("1. Simple arithmetic (ADDI)")
    # addi x1, x0, 42  (x1 = 0 + 42)
    # Encoding: opcode=0x13, rd=1, funct3=0, rs1=0, imm=42
    # 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (42 << 20)
    addi_instr = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (42 << 20)
    changes = model.execute(addi_instr)
    print(f"   Executed ADDI x1, x0, 42")
    print(f"   x1 = {model.get_gpr(1)}")
    print(f"   Changes: {model.query_changes('simple')}\n")

    # Example 2: Register-to-register operation
    print("2. Register operation (ADD)")
    # add x2, x1, x1  (x2 = x1 + x1)
    # Encoding: opcode=0x33, rd=2, funct3=0, rs1=1, rs2=1, funct7=0
    add_instr = 0x33 | (2 << 7) | (0 << 12) | (1 << 15) | (1 << 20) | (0 << 25)
    changes = model.execute(add_instr)
    print(f"   Executed ADD x2, x1, x1")
    print(f"   x1 = {model.get_gpr(1)}, x2 = {model.get_gpr(2)}")
    print(f"   Changes: {model.query_changes('simple')}\n")

    # Example 3: Speculation
    print("3. Speculation (what would happen if we execute SUB?)")
    # sub x3, x2, x1  (x3 = x2 - x1)
    # Encoding: opcode=0x33, rd=3, funct3=0, rs1=2, rs2=1, funct7=0x20
    sub_instr = 0x33 | (3 << 7) | (0 << 12) | (2 << 15) | (1 << 20) | (0x20 << 25)
    spec_changes = model.speculate(sub_instr)
    print(f"   Speculated SUB x3, x2, x1")
    print(f"   Would write x3 = {spec_changes.gpr_writes[0].value if spec_changes and spec_changes.gpr_writes else 'N/A'}")
    print(f"   Current x3 = {model.get_gpr(3)} (unchanged)")
    print(f"   Speculation changes: {model.query_changes('simple')}\n")

    # Example 4: Branch instruction
    print("4. Branch instruction (BEQ)")
    # beq x1, x2, -8  (if x1 == x2, branch back 8 bytes)
    # Encoding: opcode=0x63, funct3=0, rs1=1, rs2=2, imm=-8 (split)
    # imm bits: [12|10:5|4:1|11] = -8 = 0xFF8
    beq_instr = 0x63 | (0 << 12) | (1 << 15) | (2 << 20) | (0xFF8 << 7) | (1 << 31)
    changes = model.execute(beq_instr)
    print(f"   Executed BEQ x1, x2, -8")
    branch_info = model.get_branch_info()
    if branch_info:
        print(f"   Branch taken: {branch_info.taken}")
        print(f"   Branch target: 0x{branch_info.target:x}")
    print(f"   PC = 0x{model.get_pc():x}")
    print(f"   Changes: {model.query_changes('simple')}\n")

    # Example 5: Detailed change query
    print("5. Detailed change query")
    # addi x4, x0, 100
    addi2_instr = 0x13 | (4 << 7) | (0 << 12) | (0 << 15) | (100 << 20)
    changes = model.execute(addi2_instr)
    print(f"   Executed ADDI x4, x0, 100")
    detailed = model.query_changes("detailed")
    print(f"   Detailed changes: {detailed}\n")

    # Example 6: CSR operation
    print("6. CSR operation (CSRRW)")
    # csrrw x5, mstatus, x1  (x5 = mstatus; mstatus = x1)
    # Encoding: opcode=0x73, funct3=1, rd=5, rs1=1, imm=0x300 (mstatus)
    csrrw_instr = 0x73 | (5 << 7) | (1 << 12) | (1 << 15) | (0x300 << 20)
    changes = model.execute(csrrw_instr)
    print(f"   Executed CSRRW x5, mstatus, x1")
    print(f"   x5 = {model.get_gpr(5)}")
    print(f"   mstatus = 0x{model.get_csr('mstatus'):x}")
    print(f"   Changes: {model.query_changes('simple')}\n")

    # Example 7: Reset
    print("7. Reset state")
    model.reset()
    print(f"   After reset:")
    print(f"   x1 = {model.get_gpr(1)}")
    print(f"   x2 = {model.get_gpr(2)}")
    print(f"   PC = 0x{model.get_pc():x}\n")

    print("=== Example Complete ===")


if __name__ == "__main__":
    main()
