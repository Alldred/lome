# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Jump instruction implementations: JAL, JALR."""

from __future__ import annotations

from typing import Optional

from lome.changes import ChangeRecord, GPRRead, GPRWrite
from lome.ras import RASModel
from lome.state import State
from lome.types import OperandValues


def execute_jal(
    operand_values: OperandValues,
    state: State,
    pc: int,
    *,
    ras: Optional[RASModel] = None,
) -> ChangeRecord:
    """Execute JAL: rd = pc + 4; pc = pc + imm

    Jump to *pc + imm* and store the return address (*pc + 4*) in rd.
    Used for direct function calls and unconditional jumps.

    Parameters:
        operand_values: OperandValues with keys ``rd`` and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd and pc change.

    Example::

        # jal x1, 0x100  — jump to pc+0x100, save return addr in x1
    """
    rd = operand_values.get("rd")
    imm = operand_values.get("imm")

    return_addr = pc + 4
    target = pc + imm

    if ras is not None and rd is not None and rd in (1, 5):
        ras.push(return_addr)

    changes = ChangeRecord()
    # Write return address to rd
    old_value = state.set_gpr(rd, return_addr)
    changes.gpr_writes.append(
        GPRWrite(register=rd, value=return_addr, old_value=old_value)
    )
    # Update PC
    changes.pc_change = (target, pc)
    return changes


def execute_jalr(
    operand_values: OperandValues,
    state: State,
    pc: int,
    *,
    ras: Optional[RASModel] = None,
) -> ChangeRecord:
    """Execute JALR: rd = pc + 4; pc = (rs1 + imm) & ~1

    Jump to *(rs1 + imm)* with the LSB cleared and store the return
    address (*pc + 4*) in rd.  Used for indirect jumps and function returns.

    Parameters:
        operand_values: OperandValues with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd and pc change.

    Example::

        # jalr x0, x1, 0  — jump to address in x1 (function return)
    """
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    return_addr = pc + 4
    target = (rs1_val + imm) & ~1  # Clear LSB for alignment

    if ras is not None and rs1_idx is not None and rs1_idx in (1, 5):
        ras.pop()
    if ras is not None and rd is not None and rd in (1, 5):
        ras.push(return_addr)

    changes = ChangeRecord()
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    # Write return address to rd
    old_value = state.set_gpr(rd, return_addr)
    changes.gpr_writes.append(
        GPRWrite(register=rd, value=return_addr, old_value=old_value)
    )
    # Update PC
    changes.pc_change = (target, pc)
    return changes
