# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred. All Rights Reserved

"""State management for RISC-V functional model: GPRs, CSRs, and PC."""

from typing import Dict, Optional

from eumos import CSRDef, GPRDef, load_all_csrs, load_all_gprs


class State:
    """Manages RISC-V architectural state: GPRs, CSRs, and PC."""

    def __init__(self):
        """Initialize state with GPRs, CSRs, and PC from Eumos (installed from GitHub)."""
        self._gpr_defs: Dict[int, GPRDef] = load_all_gprs()
        self._csr_defs: Dict[str, CSRDef] = load_all_csrs()
        self._csr_by_address: Dict[int, CSRDef] = {
            csr.address: csr for csr in self._csr_defs.values()
        }

        # Initialize GPRs with reset values
        self._gprs: Dict[int, int] = {}
        for idx in range(32):
            gpr_def = self._gpr_defs.get(idx)
            reset_value = gpr_def.reset_value if gpr_def else 0
            self._gprs[idx] = reset_value

        # Initialize CSRs with reset values
        self._csrs: Dict[int, int] = {}
        for csr_def in self._csr_defs.values():
            reset_value = csr_def.reset_value if csr_def.reset_value is not None else 0
            self._csrs[csr_def.address] = reset_value

        # Initialize PC
        self._pc: int = 0

    def get_gpr(self, reg: int) -> int:
        """Get GPR value. x0 always returns 0."""
        if not (0 <= reg <= 31):
            raise ValueError(f"GPR index must be 0-31, got {reg}")
        if reg == 0:
            return 0  # x0 is always zero
        return self._gprs.get(reg, 0)

    def set_gpr(self, reg: int, value: int) -> Optional[int]:
        """Set GPR value. Returns old value. x0 writes are ignored (returns old value 0)."""
        if not (0 <= reg <= 31):
            raise ValueError(f"GPR index must be 0-31, got {reg}")
        if reg == 0:
            return 0  # x0 is read-only, ignore writes
        old_value = self._gprs.get(reg, 0)
        # Mask to 64 bits (sign-extend for 32-bit operations handled by instructions)
        self._gprs[reg] = value & 0xFFFFFFFFFFFFFFFF
        return old_value

    def get_csr(self, csr: int) -> Optional[int]:
        """Get CSR value by 12-bit address. Returns None if CSR doesn't exist."""
        addr = csr & 0xFFF
        return self._csrs.get(addr)

    def set_csr(self, csr: int, value: int) -> Optional[int]:
        """Set CSR value by 12-bit address. Returns old value or None if CSR doesn't exist."""
        addr = csr & 0xFFF
        if addr not in self._csrs:
            return None
        csr_def = self._csr_by_address.get(addr)
        if csr_def and csr_def.access == "read-only":
            # Read-only CSRs can't be written (but we still return old value for tracking)
            return self._csrs.get(addr)
        old_value = self._csrs.get(addr, 0)
        # Mask to CSR width
        if csr_def and csr_def.width:
            mask = (1 << csr_def.width) - 1
            self._csrs[addr] = value & mask
        else:
            # Default to 64 bits if width not specified
            self._csrs[addr] = value & 0xFFFFFFFFFFFFFFFF
        return old_value

    def get_csr_by_name(self, name: str) -> Optional[int]:
        """Get CSR value by name. Returns None if CSR doesn't exist."""
        csr_def = self._csr_defs.get(name)
        if csr_def is None:
            return None
        return self.get_csr(csr_def.address)

    def set_csr_by_name(self, name: str, value: int) -> Optional[int]:
        """Set CSR value by name. Returns old value or None if CSR doesn't exist."""
        csr_def = self._csr_defs.get(name)
        if csr_def is None:
            return None
        return self.set_csr(csr_def.address, value)

    def get_pc(self) -> int:
        """Get program counter."""
        return self._pc

    def set_pc(self, value: int) -> int:
        """Set program counter. Returns old value."""
        old_pc = self._pc
        self._pc = value & 0xFFFFFFFFFFFFFFFF
        return old_pc

    def reset(self) -> None:
        """Reset all state to initial values."""
        # Reset GPRs
        for idx in range(32):
            gpr_def = self._gpr_defs.get(idx)
            reset_value = gpr_def.reset_value if gpr_def else 0
            self._gprs[idx] = reset_value

        # Reset CSRs
        for csr_def in self._csr_defs.values():
            reset_value = csr_def.reset_value if csr_def.reset_value is not None else 0
            self._csrs[csr_def.address] = reset_value

        # Reset PC
        self._pc = 0

    def snapshot(self) -> Dict:
        """Create a snapshot of current state for speculation."""
        return {
            "gprs": self._gprs.copy(),
            "csrs": self._csrs.copy(),
            "pc": self._pc,
        }

    def restore(self, snapshot: Dict) -> None:
        """Restore state from a snapshot."""
        self._gprs = snapshot["gprs"].copy()
        self._csrs = snapshot["csrs"].copy()
        self._pc = snapshot["pc"]

    def get_gpr_def(self, reg: int) -> Optional[GPRDef]:
        """Get GPR definition for a register index."""
        return self._gpr_defs.get(reg)

    def get_csr_def(self, csr: int) -> Optional[CSRDef]:
        """Get CSR definition for a CSR address."""
        addr = csr & 0xFFF
        return self._csr_by_address.get(addr)

    def get_csr_def_by_name(self, name: str) -> Optional[CSRDef]:
        """Get CSR definition by name."""
        return self._csr_defs.get(name)
