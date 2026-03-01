# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Opcode builder backed by Eumos instruction encoding."""

from __future__ import annotations

from functools import lru_cache

from eumos import Eumos
from eumos.encoder import encode_instruction


@lru_cache(maxsize=1)
def _eumos() -> Eumos:
    return Eumos()


def opc(mnemonic: str, **operand_values: int) -> int:
    resolved = {k: int(v) for k, v in operand_values.items()}
    instr = _eumos().instructions[mnemonic]
    return encode_instruction(instr, resolved)
