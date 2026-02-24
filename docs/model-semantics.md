<!--
  ~ SPDX-License-Identifier: MIT
  ~ Copyright (c) 2026 Stuart Alldred.
  -->

# Model Semantics

[Docs Home](README.md) | [Floating-Point Behavior](floating-point.md)

## Architectural vs Raw Access

### Architectural (`get_*` / `set_*`)

- Enforces architectural constraints (`x0` behavior, CSR read-only handling)
- Triggers side effects where defined (CSR hooks)

### Raw (`peek_*` / `poke_*`)

- Bypasses architectural constraints
- Never triggers side effects
- Useful for test setup, debug, state injection, checkpoint restore

## PC Update Rules

After each non-speculative execution:

- If `ChangeRecord` includes an exception (`exception` or `exception_code`):
  - PC is set from MTVEC. Base = 4-byte-aligned MTVEC base; mode = MTVEC bits 1:0. **Direct mode (0)**: PC = base. **Vectored mode (1)**: synchronous exceptions → PC = base; interrupts (mcause high bit set) → PC = base + 4 × cause code. If MTVEC is not present in the ISA, PC is left unchanged.
- Else if `pc_change` exists:
  - PC is set to `pc_change[0]`
- Else:
  - PC advances by `+4`

## Speculation

`speculate()` runs the same instruction path but restores state afterward.
Returned `ChangeRecord` still reports the would-be effects.

## Exceptions and Illegal Instructions

- Unknown mnemonics become `illegal_instruction` changes with mcause code `2`.
- Handler runtime errors are surfaced as real Python exceptions (not swallowed).

## Change Tracking

Each `ChangeRecord` can include:

- register/CSR/FPR reads and writes
- `pc_change`
- memory accesses
- branch outcome (`BranchInfo`)
- exception metadata

Use:

- `to_simple_dict()` for compact summaries
- `to_detailed_dict()` for full trace data
