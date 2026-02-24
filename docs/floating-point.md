<!--
  ~ SPDX-License-Identifier: MIT
  ~ Copyright (c) 2026 Stuart Alldred.
  -->

# Floating-Point Behavior

[Docs Home](README.md) | [Instruction Support](instruction-support.md)

## Scope

Lome includes broad F/D functional support, but prioritizes functional behavior
over bit-exact softfloat-style emulation.

## Current Model

- Uses Python `float` conversions for many floating-point operations.
- Supports FPR read/write tracking through `ChangeRecord`.
- Supports many move/convert/class operations and FP load/store.

## Rounding and Flags

Current limitations:

- `fflags` exception semantics are not fully modeled.
- per-instruction rounding mode behavior is not strictly enforced for all paths.
- helper `round_for_float` currently returns the input value unchanged.

This is acceptable for many functional workflows, but not for strict IEEE 754
conformance testing.

## Practical Guidance

Use Lome FP behavior for:

- ISA flow validation
- instruction stream functional checks
- non-bit-exact model integration work

Avoid relying on it for:

- full IEEE 754 compliance validation
- precise exception-flag behavior verification
