# Calibration Auditability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve every development calibration run in deterministic,
strictly validated raw JSON before the frozen configuration is certified.

**Architecture:** A private `clocks._calibration` module owns compact schema-v1
document construction and canonical JSON I/O. Both scan scripts write
block-specific raw files. A repository archiver accepts explicit input paths,
validates the complete development Cartesian products, wraps the already-run
legacy echo JSON with source provenance, and writes only seed-block-zero data
to development-labelled tracked paths.

**Tech Stack:** Python standard-library JSON/SHA-256, NumPy array conversion,
pytest, Ruff.

---

### Task 1: Deterministic raw-study schema

**Files:**

- Create: `src/clocks/_calibration.py`
- Create: `tests/test_calibration.py`

1. Write failing tests for schema metadata, NumPy conversion, stable result
   order, `allow_nan=False`, trailing newline, and byte-identical repeated
   writes.
2. Run the focused tests and confirm failure because the module is absent.
3. Implement the minimal schema-v1 document and canonical JSON helpers.
4. Run the focused tests and confirm they pass.

### Task 2: Block-specific scan outputs

**Files:**

- Modify: `scripts/scan_multi_mass_2d.py`
- Modify: `scripts/scan_echolocation_range.py`
- Modify: `src/clocks/_echo_study.py`
- Modify: `tests/test_scenarios.py`
- Modify: `tests/test_echo_study.py`

1. Write failing synthetic tests proving each scan selects a block-specific
   path and writes schema-v1 metadata containing the actual seed/control grid,
   tolerances, ranges where applicable, and all result fields.
2. Run the tests and confirm the expected missing APIs/metadata.
3. Route both scan writers through `clocks._calibration`; do not execute a
   scan.
4. Run the focused tests and confirm they pass.

### Task 3: Strict development archiver

**Files:**

- Create: `scripts/archive_development_calibration.py`
- Modify: `tests/test_calibration.py`

1. Build synthetic complete 324-record multi and 1,944-record echo studies.
2. Write failing tests for successful canonical archival and rejection of
   nonzero blocks, mixed or missing seeds, duplicate/missing control tuples,
   wrong counts, ranges, or tolerances.
3. Add a legacy-echo test that requires schema-v1 wrapping only after full
   validation and records the exact input SHA-256.
4. Implement strict validation and deterministic writes to
   `docs/calibration/*_development.json`.
5. Run the focused tests and confirm they pass.

### Task 4: Pending report language and verification

**Files:**

- Modify: `docs/2026-08-16-development-calibration.md`

1. Correct the close-range maximum position error to 0.0633.
2. Name the future tracked development artifacts without claiming that they
   exist; retain pending auditability language until the main session runs the
   full multi grid and archiver.
3. Run focused fast tests, Ruff, prose checks, and `git diff --check` without
   running slow tests, development inference grids, or protected seeds.
