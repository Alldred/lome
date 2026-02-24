# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""System instruction implementations: CSRRW, CSRRS, CSRRC, CSRRWI, CSRRSI, CSRRCI, ECALL, EBREAK, MRET, FENCE, FENCE.TSO, LUI, AUIPC."""

from __future__ import annotations

from lome.changes import ChangeRecord, CSRRead, CSRWrite, GPRRead, GPRWrite
from lome.state import State


def execute_csrrw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute CSRRW: rd = CSR[imm]; CSR[imm] = rs1

    Atomically read the CSR into rd and write the value of rs1 into the
    CSR.  CSR Read/Write.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm`` (CSR address).
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing CSR write and GPR write.

    Example::

        # csrrw x1, mstatus, x2  — x1 = mstatus; mstatus = x2
    """
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    csr_addr = operand_values.get("imm")

    csr_val = state.get_csr(csr_addr)
    rs1_val = state.get_gpr(rs1_idx)

    changes = ChangeRecord()
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    csr_def = state.get_csr_def(csr_addr)
    csr_name = csr_def.name if csr_def else None
    if csr_val is not None:
        changes.csr_reads.append(
            CSRRead(address=csr_addr, name=csr_name, value=csr_val)
        )
    # Write CSR
    old_csr_val = state.set_csr(csr_addr, rs1_val)
    if old_csr_val is not None:
        changes.csr_writes.append(
            CSRWrite(
                address=csr_addr, name=csr_name, value=rs1_val, old_value=old_csr_val
            )
        )
    # Write old CSR value to rd
    if csr_val is not None:
        old_value = state.set_gpr(rd, csr_val)
        changes.gpr_writes.append(
            GPRWrite(register=rd, value=csr_val, old_value=old_value)
        )
    return changes


def execute_csrrs(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute CSRRS: rd = CSR[imm]; CSR[imm] = CSR[imm] | rs1

    Read the CSR into rd, then set bits in the CSR that are set in rs1.
    If rs1 is x0 this is a pure read (``csrr``).

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm`` (CSR address).
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing CSR write (if any) and GPR write.

    Example::

        # csrrs x1, mstatus, x2  — x1 = mstatus; mstatus |= x2
    """
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    csr_addr = operand_values.get("imm")

    csr_val = state.get_csr(csr_addr)
    rs1_val = state.get_gpr(rs1_idx)

    changes = ChangeRecord()
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    csr_def = state.get_csr_def(csr_addr)
    csr_name = csr_def.name if csr_def else None
    if csr_val is not None:
        changes.csr_reads.append(
            CSRRead(address=csr_addr, name=csr_name, value=csr_val)
        )
    # Update CSR (only if rs1 != 0)
    if rs1_val != 0 and csr_val is not None:
        new_csr_val = csr_val | rs1_val
        old_csr_val = state.set_csr(csr_addr, new_csr_val)
        if old_csr_val is not None:
            changes.csr_writes.append(
                CSRWrite(
                    address=csr_addr,
                    name=csr_name,
                    value=new_csr_val,
                    old_value=old_csr_val,
                )
            )
    # Write old CSR value to rd
    if csr_val is not None:
        old_value = state.set_gpr(rd, csr_val)
        changes.gpr_writes.append(
            GPRWrite(register=rd, value=csr_val, old_value=old_value)
        )
    return changes


def execute_csrrc(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute CSRRC: rd = CSR[imm]; CSR[imm] = CSR[imm] & ~rs1

    Read the CSR into rd, then clear bits in the CSR that are set in rs1.
    If rs1 is x0 this is a pure read.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm`` (CSR address).
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing CSR write (if any) and GPR write.

    Example::

        # csrrc x1, mstatus, x2  — x1 = mstatus; mstatus &= ~x2
    """
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    csr_addr = operand_values.get("imm")

    csr_val = state.get_csr(csr_addr)
    rs1_val = state.get_gpr(rs1_idx)

    changes = ChangeRecord()
    if rs1_idx is not None:
        changes.gpr_reads.append(GPRRead(register=rs1_idx, value=rs1_val))
    csr_def = state.get_csr_def(csr_addr)
    csr_name = csr_def.name if csr_def else None
    if csr_val is not None:
        changes.csr_reads.append(
            CSRRead(address=csr_addr, name=csr_name, value=csr_val)
        )
    # Update CSR (only if rs1 != 0)
    if rs1_val != 0 and csr_val is not None:
        new_csr_val = csr_val & ~rs1_val
        old_csr_val = state.set_csr(csr_addr, new_csr_val)
        if old_csr_val is not None:
            changes.csr_writes.append(
                CSRWrite(
                    address=csr_addr,
                    name=csr_name,
                    value=new_csr_val,
                    old_value=old_csr_val,
                )
            )
    # Write old CSR value to rd
    if csr_val is not None:
        old_value = state.set_gpr(rd, csr_val)
        changes.gpr_writes.append(
            GPRWrite(register=rd, value=csr_val, old_value=old_value)
        )
    return changes


def execute_csrrwi(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute CSRRWI: rd = CSR[imm]; CSR[imm] = zimm (zero-extended immediate)

    Atomically read the CSR into rd and write the zero-extended 5-bit
    immediate into the CSR.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1`` (contains zimm), and
            ``imm`` (CSR address).
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing CSR write and GPR write.

    Example::

        # csrrwi x1, mstatus, 3  — x1 = mstatus; mstatus = 3
    """
    rd = operand_values.get("rd")
    zimm = operand_values.get("rs1")  # In immediate variant, rs1 field contains zimm
    csr_addr = operand_values.get("imm")

    csr_val = state.get_csr(csr_addr)
    zimm_val = zimm & 0x1F  # Zero-extend 5-bit immediate

    changes = ChangeRecord()
    # Write CSR
    old_csr_val = state.set_csr(csr_addr, zimm_val)
    if old_csr_val is not None:
        csr_def = state.get_csr_def(csr_addr)
        csr_name = csr_def.name if csr_def else None
        changes.csr_writes.append(
            CSRWrite(
                address=csr_addr, name=csr_name, value=zimm_val, old_value=old_csr_val
            )
        )
    # Write old CSR value to rd
    if csr_val is not None:
        old_value = state.set_gpr(rd, csr_val)
        changes.gpr_writes.append(
            GPRWrite(register=rd, value=csr_val, old_value=old_value)
        )
    return changes


def execute_csrrsi(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute CSRRSI: rd = CSR[imm]; CSR[imm] = CSR[imm] | zimm

    Read the CSR into rd, then set bits in the CSR corresponding to the
    zero-extended 5-bit immediate.  No write if zimm is zero.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1`` (contains zimm), and
            ``imm`` (CSR address).
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing CSR write (if any) and GPR write.

    Example::

        # csrrsi x1, mstatus, 2  — x1 = mstatus; mstatus |= 2
    """
    rd = operand_values.get("rd")
    zimm = operand_values.get("rs1")  # In immediate variant, rs1 field contains zimm
    csr_addr = operand_values.get("imm")

    csr_val = state.get_csr(csr_addr)
    zimm_val = zimm & 0x1F  # Zero-extend 5-bit immediate

    changes = ChangeRecord()
    # Update CSR (only if zimm != 0)
    if zimm_val != 0 and csr_val is not None:
        new_csr_val = csr_val | zimm_val
        old_csr_val = state.set_csr(csr_addr, new_csr_val)
        if old_csr_val is not None:
            csr_def = state.get_csr_def(csr_addr)
            csr_name = csr_def.name if csr_def else None
            changes.csr_writes.append(
                CSRWrite(
                    address=csr_addr,
                    name=csr_name,
                    value=new_csr_val,
                    old_value=old_csr_val,
                )
            )
    # Write old CSR value to rd
    if csr_val is not None:
        old_value = state.set_gpr(rd, csr_val)
        changes.gpr_writes.append(
            GPRWrite(register=rd, value=csr_val, old_value=old_value)
        )
    return changes


def execute_csrrci(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute CSRRCI: rd = CSR[imm]; CSR[imm] = CSR[imm] & ~zimm

    Read the CSR into rd, then clear bits in the CSR corresponding to the
    zero-extended 5-bit immediate.  No write if zimm is zero.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1`` (contains zimm), and
            ``imm`` (CSR address).
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing CSR write (if any) and GPR write.

    Example::

        # csrrci x1, mstatus, 2  — x1 = mstatus; mstatus &= ~2
    """
    rd = operand_values.get("rd")
    zimm = operand_values.get("rs1")  # In immediate variant, rs1 field contains zimm
    csr_addr = operand_values.get("imm")

    csr_val = state.get_csr(csr_addr)
    zimm_val = zimm & 0x1F  # Zero-extend 5-bit immediate

    changes = ChangeRecord()
    # Update CSR (only if zimm != 0)
    if zimm_val != 0 and csr_val is not None:
        new_csr_val = csr_val & ~zimm_val
        old_csr_val = state.set_csr(csr_addr, new_csr_val)
        if old_csr_val is not None:
            csr_def = state.get_csr_def(csr_addr)
            csr_name = csr_def.name if csr_def else None
            changes.csr_writes.append(
                CSRWrite(
                    address=csr_addr,
                    name=csr_name,
                    value=new_csr_val,
                    old_value=old_csr_val,
                )
            )
    # Write old CSR value to rd
    if csr_val is not None:
        old_value = state.set_gpr(rd, csr_val)
        changes.gpr_writes.append(
            GPRWrite(register=rd, value=csr_val, old_value=old_value)
        )
    return changes


def execute_lui(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute LUI: rd = imm << 12

    Load the 20-bit immediate into the upper bits of rd (bits 31:12),
    zeroing bits 11:0 and sign-extending to 64 bits.

    Parameters:
        operand_values: dict with keys ``rd`` and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # lui x1, 0x12345  — x1 = 0x12345000
    """
    rd = operand_values.get("rd")
    imm = operand_values.get("imm")

    result = (imm << 12) & 0xFFFFFFFFFFFFFFFF

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_auipc(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute AUIPC: rd = pc + (imm << 12)

    Add the upper immediate (shifted left 12 bits) to the PC and store
    the result in rd.  Used for PC-relative addressing.

    Parameters:
        operand_values: dict with keys ``rd`` and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # auipc x1, 0x10  — x1 = pc + 0x10000
    """
    rd = operand_values.get("rd")
    imm = operand_values.get("imm")

    result = (pc + (imm << 12)) & 0xFFFFFFFFFFFFFFFF

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_ecall(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute ECALL: Environment call

    Raise an environment-call exception to request a service from the
    execution environment (e.g. OS syscall).

    Parameters:
        operand_values: dict (unused — ECALL has no operands).
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord with exception set to ``"environment_call"``.

    Example::

        # ecall  — trigger environment call (syscall)
    """
    changes = ChangeRecord()
    # Note: We do not set exception_code (mcause) here because the current
    # privilege mode (U/S/M) is not tracked at this level. A higher-level
    # component that knows the active privilege should assign the correct
    # mcause value (e.g. 8/9/10) based on this ECALL exception.
    changes.exception = "environment_call"
    # ECALL doesn't update PC in normal execution (trap handler does)
    # But we track it as a change for query purposes
    changes.pc_change = (pc + 4, pc)  # Next instruction (though trap will redirect)
    return changes


def execute_ebreak(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute EBREAK: Environment breakpoint

    Raise a breakpoint exception, typically used by debuggers.

    Parameters:
        operand_values: dict (unused — EBREAK has no operands).
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord with exception set to ``"breakpoint"``.

    Example::

        # ebreak  — trigger breakpoint for debugger
    """
    changes = ChangeRecord()
    changes.exception_code = 3  # RISC-V mcause: Breakpoint
    changes.exception = "breakpoint"
    # EBREAK doesn't update PC in normal execution (debugger handles it)
    changes.pc_change = (pc + 4, pc)  # Next instruction (though debugger will redirect)
    return changes


def execute_mret(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute MRET: Return from machine mode trap handler

    Restore the PC from the ``mepc`` CSR and return to the interrupted
    context.

    Parameters:
        operand_values: dict (unused — MRET has no operands).
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the pc change (jump to mepc).

    Example::

        # mret  — return from M-mode trap, pc = mepc
    """
    # MRET restores PC from mepc CSR
    mepc = state.get_csr_by_name("mepc")
    target = mepc if mepc is not None else pc + 4

    changes = ChangeRecord()
    changes.pc_change = (target, pc)
    return changes


def execute_fence(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute FENCE: Memory and I/O ordering

    Ensure that all memory and I/O operations before the fence are
    observed before those after it.  This is a no-op in the functional
    model since memory ordering is not modeled.

    Parameters:
        operand_values: dict (fence fields are not currently decoded).
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord (empty — no architectural state changes).

    Example::

        # fence rw, rw  — order prior reads/writes before subsequent ones
    """
    # FENCE is a no-op in functional model (ordering is not modeled)
    changes = ChangeRecord()
    return changes


def execute_fence_tso(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute FENCE.TSO: Total Store Ordering

    A stricter fence that enforces total-store-order semantics.  This is
    a no-op in the functional model since memory ordering is not modeled.

    Parameters:
        operand_values: dict (unused).
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord (empty — no architectural state changes).

    Example::

        # fence.tso  — enforce total store ordering
    """
    # FENCE.TSO is a no-op in functional model (ordering is not modeled)
    changes = ChangeRecord()
    return changes
