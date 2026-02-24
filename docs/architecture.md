<!--
  ~ SPDX-License-Identifier: MIT
  ~ Copyright (c) 2026 Stuart Alldred.
  -->

# Architecture

[Docs Home](README.md) | [Model Semantics](model-semantics.md)

## High-Level Components

- `Lome` (`lome/model.py`): main public API
- `State` (`lome/state.py`): GPR/FPR/CSR/PC storage and access semantics
- `Decoder` (from Eumos): decodes instruction words into instruction instances
- `Executor` (`lome/executor.py`): dispatches decoded mnemonics to handlers
- Instruction handlers (`lome/instructions/*.py`): per-instruction behavior
- `ChangeRecord` (`lome/changes.py`): records reads/writes/pc/memory/branch/exception

## Execution Flow

1. `Lome.execute()` normalizes instruction bytes/int and reads current PC.
2. Eumos decoder converts the word into an instruction instance.
3. Executor dispatches to a mnemonic-specific handler.
4. Handler updates state and returns a `ChangeRecord`.
5. `Lome.execute()` updates PC using change record semantics.

## Dispatch Model

The executor keeps a static mnemonic → handler map for predictable dispatch.
Optional dependencies are injected only for relevant mnemonics:

- `memory` for load/store instructions
- `ras` for `jal` / `jalr`

## State Design

- GPR and FPR storage are fixed-size list-backed structures (fast indexed access).
- CSR storage is address-keyed dictionary-backed (sparse by ISA definition).
- CSR write hooks model side effects (`mstatus`/`sstatus` mirroring by default).

## Serialization Boundaries

- `export_state()` and `restore_state()` are raw state transfers.
- Restores intentionally bypass architectural side effects.

## Relationship to Eumos

Lome is driven by Eumos definitions and decode behavior:

- instruction support follows Eumos instruction set content
- CSR/GPR/FPR definitions and reset values come from Eumos
