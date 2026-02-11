# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Example usage of RISC-V functional model.

Demonstrates:
  1. Loading Eumos once and passing to the model
  2. Simple arithmetic (ADDI, ADD)
  3. Peek / poke for state setup
  4. Speculation (dry-run execution)
  5. Branch instructions
  6. Change tracking (simple and detailed dicts)
  7. CSR operations (CSRRW)
  8. JSON export / restore round-trip
  9. Reset
"""

import sys
from pathlib import Path

# Add parent directory to path (for running as a script)
sys.path.insert(0, str(Path(__file__).parent.parent))

from eumos import Eumos

from riscv_model import RISCVModel


def main():
    """Run all examples."""

    print("=== RISC-V Functional Model Examples ===\n")

    # ------------------------------------------------------------------
    # 0. Load Eumos once, share across all models
    # ------------------------------------------------------------------
    print("0. Load Eumos (once)")
    isa = Eumos()
    print(f"   Loaded {len(isa.gprs)} GPR defs, {len(isa.csrs)} CSR defs\n")

    model = RISCVModel(isa)

    # ------------------------------------------------------------------
    # 1. Simple arithmetic
    # ------------------------------------------------------------------
    print("1. Simple arithmetic (ADDI)")
    addi_instr = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (42 << 20)
    changes = model.execute(addi_instr)
    print("   Executed: ADDI x1, x0, 42")
    print(f"   x1 = {model.get_gpr(1)}")
    print(f"   Changes: {changes.to_simple_dict()}\n")

    # ------------------------------------------------------------------
    # 2. Peek / poke for state setup
    # ------------------------------------------------------------------
    print("2. Peek / poke (raw state manipulation)")
    model2 = RISCVModel(isa)
    model2.poke_gpr(1, 0x1000)
    model2.poke_gpr(2, 0x2000)
    model2.poke_pc(0x8000_0000)
    model2.poke_csr(0x300, 0xFF)
    print(f"   poke_gpr(1, 0x1000) -> get_gpr(1) = 0x{model2.get_gpr(1):x}")
    print(f"   poke_pc(0x80000000) -> get_pc()   = 0x{model2.get_pc():x}")
    print(f"   poke_csr(0x300, 0xFF) -> peek_csr(0x300) = 0x{model2.peek_csr(0x300):x}")

    model2.poke_gpr(0, 0xDEAD)
    print(f"   poke_gpr(0, 0xDEAD) -> peek_gpr(0) = 0x{model2.peek_gpr(0):x}")
    print(
        f"                          get_gpr(0)  = {model2.get_gpr(0)}  (architectural: always 0)\n"
    )

    # ------------------------------------------------------------------
    # 3. Register-to-register operation
    # ------------------------------------------------------------------
    print("3. Register operation (ADD)")
    add_instr = 0x33 | (2 << 7) | (0 << 12) | (1 << 15) | (1 << 20) | (0 << 25)
    changes = model.execute(add_instr)
    print("   Executed: ADD x2, x1, x1")
    print(f"   x1 = {model.get_gpr(1)}, x2 = {model.get_gpr(2)}")
    print(f"   Changes: {changes.to_simple_dict()}\n")

    # ------------------------------------------------------------------
    # 4. Speculation
    # ------------------------------------------------------------------
    print("4. Speculation (what would SUB do?)")
    sub_instr = 0x33 | (3 << 7) | (0 << 12) | (2 << 15) | (1 << 20) | (0x20 << 25)
    spec_changes = model.speculate(sub_instr)
    would_write = spec_changes.gpr_writes[0].value if spec_changes.gpr_writes else "N/A"
    print("   Speculated: SUB x3, x2, x1")
    print(f"   Would write x3 = {would_write}")
    print(f"   Current x3 = {model.get_gpr(3)} (unchanged)")
    print(f"   Speculation changes: {spec_changes.to_simple_dict()}\n")

    # ------------------------------------------------------------------
    # 5. Branch instruction
    # ------------------------------------------------------------------
    print("5. Branch instruction (BEQ)")
    beq_instr = 0x63 | (0 << 12) | (1 << 15) | (2 << 20) | (0xFF8 << 7) | (1 << 31)
    changes = model.execute(beq_instr)
    print("   Executed: BEQ x1, x2, -8")
    if changes and changes.branch_info:
        print(f"   Branch taken: {changes.branch_info.taken}")
        print(f"   Branch target: 0x{changes.branch_info.target:x}")
    print(f"   PC = 0x{model.get_pc():x}")
    print(f"   Changes: {changes.to_simple_dict()}\n")

    # ------------------------------------------------------------------
    # 6. get_changes() after execute
    # ------------------------------------------------------------------
    print("6. get_changes() after execute")
    addi2_instr = 0x13 | (4 << 7) | (0 << 12) | (0 << 15) | (100 << 20)
    model.execute(addi2_instr)
    last = model.get_changes()
    print("   Executed: ADDI x4, x0, 100")
    print(f"   Detailed changes: {last.to_detailed_dict()}\n")

    # ------------------------------------------------------------------
    # 7. CSR operation
    # ------------------------------------------------------------------
    print("7. CSR operation (CSRRW)")
    csrrw_instr = 0x73 | (5 << 7) | (1 << 12) | (1 << 15) | (0x300 << 20)
    changes = model.execute(csrrw_instr)
    print("   Executed: CSRRW x5, mstatus, x1")
    print(f"   x5 = {model.get_gpr(5)}")
    mstatus_val = model.get_csr("mstatus")
    print(
        f"   mstatus = 0x{mstatus_val:x}"
        if mstatus_val is not None
        else "   mstatus = None"
    )
    print(f"   Changes: {changes.to_simple_dict()}\n")

    # ------------------------------------------------------------------
    # 8. JSON export / restore
    # ------------------------------------------------------------------
    print("8. JSON export / restore")
    data = model.export_state()
    json_str = model.export_state_json(indent=2)
    print(f"   Exported state ({len(json_str)} chars JSON)")
    print(f"   PC = {data['pc']}")
    print(f"   x1 = {data['gprs']['1']}")

    restored = RISCVModel(isa)
    restored.restore_state(data)
    print(f"   Restored: x1 = {restored.get_gpr(1)}, PC = 0x{restored.get_pc():x}")

    model_from_json = RISCVModel.from_json(json_str, isa)
    print(f"   from_json: x1 = {model_from_json.get_gpr(1)}\n")

    # ------------------------------------------------------------------
    # 9. Reset
    # ------------------------------------------------------------------
    print("9. Reset state")
    model.reset()
    print("   After reset:")
    print(f"   x1 = {model.get_gpr(1)}")
    print(f"   x2 = {model.get_gpr(2)}")
    print(f"   PC = 0x{model.get_pc():x}\n")

    print("=== Examples Complete ===")


if __name__ == "__main__":
    main()
