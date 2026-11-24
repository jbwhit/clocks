# Replay Clips Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw the six demo animations when the site is built, as H.264 clips with posters in both themes, from the committed replay archives; switch the demo commands, the pages and the provenance prose to them; and remove the committed GIFs with the code that made them.

**Architecture:** (skeleton; filled in by the following commits)

**Tech Stack:** Python 3.12+, NumPy, Matplotlib 3.10, FFmpeg through `imageio-ffmpeg` (development dependency), pytest, uv, ruff, Quarto 1.11.9 on the runner, GitHub Actions.

**Spec:** docs/plans/2026-11-19-replay-and-animation-design.md

**Status:** Draft skeleton. Not yet reviewed.

## Global Constraints

(to be written)

## Review Focus

(to be written)

## Decisions this plan makes

(to be written)

## Measurements

(to be written)

---

Task list (provisional):

1. Probe: every clip drawn and timed; Quarto 1.11.9 clean build; cold-runner timings; the format decision.
2. `_video.py`: the dwell schedule, the encoder, the decode check.
3. Clip builders.
4. The demo commands, public names, version and upgrade note.
5. The 2D truth change and the box-margin guard.
6. The pre-render script, embeds and descriptions.
7. Clip support in the rendered-site check and the source guard.
8. Provenance prose.
9. Removal of the GIFs, `_animate.py`, the old panel functions and their tests.
10. Mutation run.
