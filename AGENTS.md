# Agent Skill Index

This repository keeps operational agent guidance in skills under `.agents/skills/`.

## Primary Skills
- `flashinfer-local-workflow`
  - File: `.agents/skills/flashinfer-local-workflow/SKILL.md`
  - Scope: local benchmark workflow, dataset conventions, NCU usage, and GDN experiment loop.
- `cute-dsl`
  - File: `.agents/skills/cute-dsl/SKILL.md`
  - Scope: CuTe DSL API/limitations/debug workflow for kernel development.

## Usage
- When a task is about FlashInfer benchmark operations in this repo, load:
  - `.agents/skills/flashinfer-local-workflow/SKILL.md`
- When a task needs CuTe DSL correctness/perf debugging, additionally load:
  - `.agents/skills/cute-dsl/SKILL.md`
