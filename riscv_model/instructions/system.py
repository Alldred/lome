# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""System instruction implementations: CSRRW, CSRRS, CSRRC, CSRRWI, CSRRSI, CSRRCI, ECALL, EBREAK, MRET, FENCE, FENCE.TSO, LUI, AUIPC."""

from riscv_model.changes import ChangeRecord, CSRWrite, GPRWrite
from riscv_model.state import State


def execute_csrrw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute CSRRW: rd = CSR[imm]; CSR[imm] = rs1"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    csr_addr = operand_values.get("imm")

    csr_val = state.get_csr(csr_addr)
    rs1_val = state.get_gpr(rs1_idx)

    changes = ChangeRecord()
    # Write CSR
    old_csr_val = state.set_csr(csr_addr, rs1_val)
    if old_csr_val is not None:
        csr_def = state.get_csr_def(csr_addr)
        csr_name = csr_def.name if csr_def else None
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
    """Execute CSRRS: rd = CSR[imm]; CSR[imm] = CSR[imm] | rs1"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    csr_addr = operand_values.get("imm")

    csr_val = state.get_csr(csr_addr)
    rs1_val = state.get_gpr(rs1_idx)

    changes = ChangeRecord()
    # Update CSR (only if rs1 != 0)
    if rs1_val != 0 and csr_val is not None:
        new_csr_val = csr_val | rs1_val
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


def execute_csrrc(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute CSRRC: rd = CSR[imm]; CSR[imm] = CSR[imm] & ~rs1"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    csr_addr = operand_values.get("imm")

    csr_val = state.get_csr(csr_addr)
    rs1_val = state.get_gpr(rs1_idx)

    changes = ChangeRecord()
    # Update CSR (only if rs1 != 0)
    if rs1_val != 0 and csr_val is not None:
        new_csr_val = csr_val & ~rs1_val
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


def execute_csrrwi(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute CSRRWI: rd = CSR[imm]; CSR[imm] = zimm (zero-extended immediate)"""
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
    """Execute CSRRSI: rd = CSR[imm]; CSR[imm] = CSR[imm] | zimm"""
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
    """Execute CSRRCI: rd = CSR[imm]; CSR[imm] = CSR[imm] & ~zimm"""
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
    """Execute LUI: rd = imm << 12"""
    rd = operand_values.get("rd")
    imm = operand_values.get("imm")

    result = (imm << 12) & 0xFFFFFFFFFFFFFFFF

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_auipc(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute AUIPC: rd = pc + (imm << 12)"""
    rd = operand_values.get("rd")
    imm = operand_values.get("imm")

    result = (pc + (imm << 12)) & 0xFFFFFFFFFFFFFFFF

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_ecall(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute ECALL: Environment call"""
    changes = ChangeRecord()
    changes.exception = "environment_call"
    # ECALL doesn't update PC in normal execution (trap handler does)
    # But we track it as a change for query purposes
    changes.pc_change = (pc + 4, pc)  # Next instruction (though trap will redirect)
    return changes


def execute_ebreak(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute EBREAK: Environment breakpoint"""
    changes = ChangeRecord()
    changes.exception = "breakpoint"
    # EBREAK doesn't update PC in normal execution (debugger handles it)
    changes.pc_change = (pc + 4, pc)  # Next instruction (though debugger will redirect)
    return changes


def execute_mret(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute MRET: Return from machine mode trap handler"""
    # MRET restores PC from mepc CSR
    mepc = state.get_csr_by_name("mepc")
    target = mepc if mepc is not None else pc + 4

    changes = ChangeRecord()
    changes.pc_change = (target, pc)
    return changes


def execute_fence(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute FENCE: Memory and I/O ordering"""
    # FENCE is a no-op in functional model (ordering is not modeled)
    changes = ChangeRecord()
    return changes


def execute_fence_tso(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute FENCE.TSO: Total Store Ordering"""
    # FENCE.TSO is a no-op in functional model (ordering is not modeled)
    changes = ChangeRecord()
    return changes
