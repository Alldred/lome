# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Main RISC-V functional model interface."""

from typing import Optional, Union

from eumos.decoder import Decoder

from riscv_model.changes import ChangeRecord
from riscv_model.executor import execute_instruction
from riscv_model.state import State


class RISCVModel:
    """RISC-V functional model for instruction execution, speculation, and change tracking."""

    def __init__(self):
        """Initialize the RISC-V model with state and decoder (Eumos from GitHub)."""
        self._state = State()
        self._decoder = Decoder()
        self._last_changes: Optional[ChangeRecord] = None

    def execute(
        self, instruction_bytes: Union[int, bytes], speculate: bool = False
    ) -> Optional[ChangeRecord]:
        """Execute an instruction.

        Args:
            instruction_bytes: 32-bit instruction as int or bytes (little-endian)
            speculate: If True, execute without modifying state

        Returns:
            ChangeRecord with all changes, or None if instruction is unknown
        """
        # Convert bytes to int if needed
        if isinstance(instruction_bytes, bytes):
            if len(instruction_bytes) < 4:
                return None
            word = int.from_bytes(instruction_bytes[:4], byteorder="little")
        else:
            word = instruction_bytes & 0xFFFFFFFF

        # Decode instruction
        pc = self._state.get_pc()
        instance = self._decoder.decode(word, pc=pc)

        # Execute instruction
        changes = execute_instruction(instance, self._state, pc, speculate=speculate)

        # Update PC if not speculating and PC changed
        if not speculate and changes and changes.pc_change:
            new_pc, _ = changes.pc_change
            self._state.set_pc(new_pc)
        elif not speculate and changes and not changes.pc_change:
            # Normal instruction advances PC by 4
            self._state.set_pc(pc + 4)

        self._last_changes = changes
        return changes

    def speculate(self, instruction_bytes: Union[int, bytes]) -> Optional[ChangeRecord]:
        """Execute instruction in speculation mode (no state changes).

        Args:
            instruction_bytes: 32-bit instruction as int or bytes

        Returns:
            ChangeRecord showing what would change
        """
        return self.execute(instruction_bytes, speculate=True)

    def get_gpr(self, reg: int) -> int:
        """Get GPR value. x0 always returns 0."""
        return self._state.get_gpr(reg)

    def get_csr(self, csr: Union[int, str]) -> Optional[int]:
        """Get CSR value by address (int) or name (str)."""
        if isinstance(csr, str):
            return self._state.get_csr_by_name(csr)
        return self._state.get_csr(csr)

    def get_pc(self) -> int:
        """Get program counter."""
        return self._state.get_pc()

    def set_pc(self, value: int) -> None:
        """Set program counter."""
        self._state.set_pc(value)

    def reset(self) -> None:
        """Reset all state to initial values."""
        self._state.reset()
        self._last_changes = None

    def get_changes(self) -> Optional[ChangeRecord]:
        """Get change record from last execution."""
        return self._last_changes

    def get_branch_info(self):
        """Get branch information from last execution, or None."""
        if self._last_changes and self._last_changes.branch_info:
            return self._last_changes.branch_info
        return None
