# Replay Commands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the seven `demo-*` commands record a run, print it from the replay and write an archive and a PNG, with the clip on request; then remove the GIF animators, the panel functions and their public names, and release the result as version 0.2.0 with an upgrade note.

**Architecture:** `clocks._demos._common` gains one writer (`write_outputs`), which stages every output of a run beside its destination and moves nothing into place until all are staged, and one parser factory, one runner and two printers shared by the seven commands. The six clip commands draw their PNG with `clocks._clips.final_frame` and their MP4 with `clocks._clips.render_clip`; `demo-density` keeps `clocks._figures.density_figure`. Before any of that, `clocks._video.encode` stops trusting an encoder that exits 0 after a broken pipe, and its count becomes, in name as in fact, the frames sent with no broken pipe seen; the commands decode a clip before publishing it. With no caller left, `_animate.py`, `_panels.py`, `_panels3d.py` and `tests/test_viz.py` are deleted and `clocks.viz` exports nine names.

**Tech Stack:** Python 3.12+, NumPy, Matplotlib 3.10, FFmpeg through `imageio-ffmpeg` (a development dependency) or `ffmpeg` on `PATH`, Pillow (development), pytest, uv, ruff, Quarto 1.11.9 on the runner, GitHub Actions.

**Spec:** docs/plans/2026-11-19-replay-and-animation-design.md (section 7; section 9 as it concerns the commands; section 11 item 3's command and removal parts; "Testing", "Site", the `tests/test_cli.py` line; "Not doing").

**Status:** Draft for review. Revised after Codex xhigh round 1 at 8f9b6f4 (NOT YET: two Blocking, five Important, three Minor) round 2 at 887bc21 (NOT YET: one Blocking, three Important, one Minor), round 3 at 180738c (NOT YET: two wording fixes) and round 4 at b643965 (NOT YET: two wording fixes). Round 5: Codex SOUND ENOUGH TO IMPLEMENT at 165c201; Gemini NOT YET (one Blocking, confirmed); revised.

This is pull request 4. It was split from pull request 3 (`docs/plans/2026-11-24-replay-clips.md`) by the coordinator's ruling during that plan's review: round 1 (Codex xhigh at `1c3a6b5`) found two Blocking faults in what is now this plan, fixed as decisions C4 and C6 below; round 2 (at `906eb19`) found that a publication spy would have counted the renames `save_replay` and the encoder make while staging; round 3 (at `c292181`) raised nothing further here. The earlier draft was written against pull request 3's plan; this revision was written against the merged tree at `6e0527c`, and every statement below about present code was read there or measured (Measurements). Codex xhigh's round 1 on this revision (at `8f9b6f4`) found that the encoder could still publish a short clip through a buffered close, and that the print test rejected NumPy's wrapped output; both are fixed (Task 1 and decision C2; Task 3), with its five Important and three Minor findings applied as the coordinator ruled, apart from the request to drop the session trailer from commits, which the project's attribution policy requires. The plan adopts pull request 3's rules for the mutation run (its Task 14 and decisions D47 and D48) and its content-identity freshness for site media (`src/clocks/_site_media.py`, which replaced its D15), which this plan does not touch.

## Global Constraints

- Python only through uv: `uv run pytest`, `uv run ruff format .`, `uv run ruff check .`, `uv run python ...`. Never bare `python` or `pip`.
- **Tasks 1 to 6 run the same cycle** (Task 7 is the mutation run): write the tests of Step 1; run them and see them fail for the stated reason; implement; run them and see them pass; run the full local gate (`uv run ruff format --check .`, `uv run ruff check .`, `uv run pytest -q`, `uv run pytest -q -m slow`) and read its output; then, in a separate command, commit with the task's message and `git push origin claude-replay-commands`. Do not open a pull request (the coordinator does). Begin every git step with an explicit `cd` to the worktree, and run `git diff --check` before each commit.
- **A task that changes a page renders the site and runs both checks** after the gate and before the commit (Tasks 3 and 5; Task 3 changes three pages, among them one edge of the Architecture diagram): `cd site && uv run --frozen quarto render && cd ..`, then `uv run --frozen python scripts/check_site_links.py` and `uv run --frozen python scripts/check_site_figures.py`, which must print `Figure contract OK: 10 pages, 18 figures checked.` (the line CI's site check printed on `ca132e1`, pull request 3's last head). State the local `quarto --version` in the commit body.
- **A task that changes the commands or the package runs the wheel check** (Tasks 3 to 6): `rm -rf dist && uv build`, then `uv run python scripts/check_wheel_entrypoints.py dist/clocks-<version>-py3-none-any.whl python3`, naming the one wheel by its exact file name (`0.1.0` until Task 6, `0.2.0` from it). `dist/` is git-ignored.
- Commit messages are passed with a quoted heredoc (`<<'MSG'`) and end with a `Co-Authored-By:` trailer naming the model that wrote the commit.
- Implementers never use `git checkout`, `git restore` or `git stash` to undo anything: a wrong edit is undone by editing. Task 7's mutants are run by the coordinator's harness in separate worktrees.
- **A red step fails on behaviour, not on a missing name.** Where a task's tests import or patch names that do not exist yet, its Step 2 adds them with bodies that do nothing useful, as the step says; where a red run against today's code would stop before any assertion (Task 3: today's parser rejects `--output-dir`), the step says so and stages the mechanical part first.
- **The repository is public.** No committed file may contain an absolute local path, a person's name or an email address; the one exception is the `Co-Authored-By:` trailer, which names a model. Test fixtures use relative stand-in paths.
- **A long job runs in the foreground** with `timeout: 600000`; if it needs longer, wait in a bounded foreground loop, `until ! pgrep -f '[p]ytest -q' >/dev/null; do sleep 10; done; tail -20 <log>`, and call again; never end a turn on a background job. The bracket in the pattern is load-bearing: the harness runs each call as `zsh -c "<command>"`, so an unbracketed pattern matches the loop's own shell and never exits.
- **Commands** (design, section 7): `uv run demo-2d` writes `output/demo_2d.npz` and `output/demo_2d.png`; `--video` adds `output/demo_2d.mp4`; `--output-dir` replaces `--output`; `--theme` is new; `demo-density` has no `--video`; the PNG is the final frame (for `demo-density`, the density figure) and needs only Matplotlib; `--video` needs an encoder, which `uv sync` provides in a checkout, and without one the command stops and says how to get one. The version becomes 0.2.0.
- **Removed from `clocks.viz`** (section 7): the six `animate_*` functions, the eleven `plot_*` functions and the two `create_*_dashboard` functions.
- **Nothing computed is compared exactly across machines, and no encoded bytes are compared.** Printed numbers are compared with lines the test formats in the same process from the same arrays; a decoded clip is checked by `probe` (codec, size, frame count), never by its bytes or size; the only pixel compared exactly is a PNG's corner, which is the figure's face colour, written without antialiasing (measured, K8).
- **Every guard pins a form.** The command list and the upgrade note are each checked as one exact text, the Architecture page's changed part as one exact subgraph and one exact set of edges, and the README's module list as one exact tuple of entries, not by a list of rules a new spelling could slip past (decision C14).
- **Atomic writes, tested from an existing output set.** Every test of a failure starts with an archive, a PNG and a clip already in place and checks all three afterwards.
- **Two tests start from a missing output directory** (`test_write_outputs_writes_the_archive_and_png_into_a_new_directory` and `test_the_default_run_writes_into_output`); every other test that writes outputs writes into a directory that exists (decision C15).
- **No archive is re-recorded and no page other than three is changed.** `git diff --stat main -- assets site/index.qmd site/story site/method site/_quarto.yml site/replay.css` is empty at the end; the pages changed are `site/reproduce/getting-started.qmd`, `site/reproduce/reproducibility.qmd` and `site/reproduce/architecture.qmd`.

## Review Focus

1. **`--output` still parses.** `--output` is an unambiguous prefix of `--output-dir`, and argparse accepts prefixes by default: a parser that merely drops the option still takes `--output x` and writes into a directory called `x` (measured, K1). Task 3.
2. **A run that fails part-way over an existing set of outputs.** An encoder that dies after the archive and PNG are staged must not leave a new archive and PNG beside an old clip; a failed PNG save must not cut short the PNG that was there; a failure of any kind that raises, a failed rename among them, must leave no staged file the handler could remove (cleanup is best-effort: a removal that fails is a note on the error, and a hard kill removes nothing, decision C6); a directory at an output name must stop the run before anything is replaced; a closed standard output must not stop publication part-way. Task 2.
3. **An older clip beside a run made without `--video`.** The clip then belongs to an earlier run than the archive and PNG beside it. It is kept, and the command says so in one line. Task 2.
4. **An encoder that exits 0 after a broken pipe, and a clip that lacks frames.** `encode` today moves whatever it wrote into place (measured, K6: the old clip replaced by five bytes), and since `stdin` is buffered, the last frames can be lost at the close, after the generator is exhausted, with the broken pipe swallowed there too (Codex, round 1). The site's clips are decoded by the rendered-site check before they ship; a command's clip is not. So `encode` refuses status 0 after any broken pipe (Task 1), and the commands decode the staged clip with `probe` and compare its frame count with the frames sent before publishing it (Task 2).
5. **A wheel installed with no encoder anywhere.** The default command must work and `--video` must fail with instructions, with `PATH`, `PYTHONPATH` and `IMAGEIO_FFMPEG_EXE` unable to supply an encoder or the checkout by inheritance; on the machine measured, the inherited `PATH` held an `ffmpeg` (K7). Task 4.
6. **Links and leftovers at the output names.** A destination that is a symbolic link is replaced, not written through; a staged file left by an interrupted run, a link among them, neither stops the next run nor is written through; a dangling link at `<demo>.mp4` counts as an earlier clip for the `Kept:` line. Task 2.
7. **`import clocks.viz` after the deletions.** Nothing may still import `_animate`, `_panels` or `_panels3d`: the demo modules stop in Task 3 (and a test keeps them off the facade), and Task 5 deletes the modules only after a search built from the whole list of removed names. Task 5.

## Decisions this plan makes

Each is something the design leaves open or states in a way that conflicts with the code; the reason follows, then the cost if wrong.

- **C1. A fourth pull request.** The design's section 11 has three pull requests and its section 7 says "Pull request 3 bumps the version to 0.2.0 and adds a short 'Upgrading from 0.1' note". Both moved here when pull request 3 was split. Pull request 3 left two pinned sentences saying the commands still write GIFs (its D43: `tests/test_generated_assets.py`, `test_readme_describes_the_site_and_the_commands_as_they_are` and `test_the_commands_are_described_as_they_are`); Task 3 replaces both, with their assertions, in the commit that makes them false.
- **C2. `encode` is fixed first, and the commands decode before publishing** (Task 1; Task 2; the coordinator's ruling on round 1's Blocking 1). `--video` publishes a clip under the user's own file name. A broken pipe, on a write or on the close that flushes `stdin`'s buffer, now reaches `_run_encoder` as `_feed`'s return value, and an encoder that exits 0 after one is an `EncoderError`; `_run_encoder` keeps its signature, so the stand-ins of pull request 3's D41 stand. Then `write_outputs` checks the staged clip with `probe`, which runs the same encoder binary as a decoder and so adds no dependency: its decoded frame count must equal the count `render_clip` sent, or the clip is not published. That is the strong claim, every frame in the file, and it is the only frame-count guarantee for a published command clip; `encode`'s own claim is only that every frame was sent with no broken pipe seen (Codex, rounds 2 and 3). What the code rejects, then, is two things: `encode` rejects an observed broken pipe (an encoder that stops reading and exits 0 without one being seen gets through `encode`), and the command's decode check rejects a frame-count mismatch, or a clip that does not decode, before publication. The cleanup of `encode`'s temporary file no longer replaces the error it is cleaning up after (it adds a note, as `_write_archive` in `clocks._replay` does). Cost: one decode per `--video` run (a fraction of a second at two observations, K5), and a clip from an encoder that exits 0 after a broken pipe is refused even if its file happened to be whole (no such encoder is known).
- **C3. The writer, then the commands, then the removals.** Task 2 adds `write_outputs` beside the present commands; Task 3 switches all seven commands and their prose at once; Task 5 deletes the old modules once nothing imports them. Each commit imports cleanly and every sentence about the commands is true at it.
- **C4. Long options cannot be abbreviated** (round 1, Blocking 2). Every command's parser is made by `make_parser`, with `allow_abbrev=False`, so `--output` is an error, as the design's "replaces" requires, and so is `--obs`. Cost: a user who abbreviated `--observations` must spell it.
- **C5. `--video` with no encoder stops before inference**, through `parser.error`: exit status 2, the library's own message (it names `imageio-ffmpeg` and `ffmpeg on PATH`), nothing written, no directory made. The design says the command "stops and says how to get one" without saying when. An `EncoderError` during the run ends the command with status 1 and a one-line message, not a traceback. Other errors (a full disk) propagate as they do today.
- **C6. Outputs are staged, then published together** (the coordinator's ruling on round 1's Blocking 3). Each output of a run is first written beside its destination under a staged name, the final name with `.staged-` in front: the archive with `save_replay`, the PNG with `save_png` into a file created exclusively, the clip with `render_clip`. Before staging, the run unlinks its own staged names (two, or three with `--video`), so a leftover of an interrupted run (a file or a link) neither blocks it nor is written through; an unlink that fails (a directory at a staged name, say) raises, before anything is written. A final name that is a directory (not a link) stops the run before anything is staged, since its rename would fail after the archive's had succeeded (Codex, round 1). Nothing is moved into place until every output of the run is staged and the clip has decoded to every frame (C2); a failure before then, when it raises through `write_outputs`, leaves every existing file as it was, and the handler tries to remove the staged files (the limits are below). Publication is one `os.replace` per file, archive, PNG, clip, inside the same cleanup, and the `Saved:` lines are printed only after the last rename, so a closed standard output cannot stop it part-way; a destination that is a symbolic link is replaced, never written through. The output directory itself may be a link: it is the user's to choose, and nothing outside the run's own names is removed. Stated limits: the two or three renames are not one atomic step, so a failure or an interruption between them (which, with the preflight, needs the file system itself to fail, or a signal) can leave files of two runs side by side. When the failure is an exception that unwinds through `write_outputs`, its handler tries to remove the staged files left and records any removal that fails as a note on the error, and the command exits with that error (`test_a_failed_rename_removes_the_staged_files` pins this); a process killed outright removes nothing, and the next run unlinks its own staged names before it stages (a `.staged-<demo>.mp4` left by a killed `--video` run stays until the next `--video` run, since a run without one stages no clip). The `Kept:` line is printed only for the six clip demos: `demo-density` has no clip and no `--video` (Gemini, round 5). An `EncoderError`'s notes (a file left behind) are printed by `run_command` after its message, one `note:` line each. Two runs at once into one directory are not supported.
- **C7. A run without `--video` keeps an existing `<demo>.mp4` and says so** (the coordinator's ruling): one line, `Kept: <path> is from an earlier run and was not updated (--video writes a new one)`, whenever an entry is at that name, a dangling link included (`exists()` alone misses one). Cost: a stale clip can sit beside a newer archive, disclosed only at run time and in the README.
- **C8. `demo-density` writes its archive too**, with `--output-dir` and `--theme` and no `--video`. Section 7 says each command records its run and writes an archive and a PNG. Pull request 2's tests that it "writes only its PNG" at a path given go (`test_density_command_writes_png_bytes_at_the_path_given`, `test_density_command_writes_only_its_png`); the per-demo command test covers what it writes now.
- **C9. The old names go at once**, with no deprecation period: section 7 calls it a documented break. None is in `clocks.__all__` (read).
- **C10. The upgrade note describes the whole release, 0.1 to 0.2.0**, not this pull request alone: of the four names it lists as added, `load_replay`, `density_figure` and `range_study_figure` arrived in pull request 2 and `render_clip` in pull request 3.
- **C11. The commands print one more decimal than the trace headers** (`trace_decimals(prior range) + 1`: three for positions and four for masses in every clip demo, K10), because review of a re-recording rests on the printed estimates (design, "Risks"). Names are printed in ASCII (`x₁` as `x1`); `demo-density`'s printing is left exactly as pull request 2 pinned it, so `ascii_name` needs only the two subscripts. Output is ASCII apart from `±` and `→`, as today. The model-comparison command records once and prints its posteriors from the replay (today it runs the comparison twice, once through `infer` for the printout and once for the GIF).
- **C12. The wheel check stays a script; its environment is a function with a unit test.** Installing a wheel is not a unit test's job: CI's "Smoke-test installed entry points" step runs the script on every run, and Tasks 3 to 6 run it by hand. What makes its "no encoder" run mean anything is the environment it builds (K7), so that is `bare_environment`, tested exactly, with mutants (Task 7). The runs themselves have no mutant; the behaviour they check (the commands with and without an encoder) has unit tests that do. CI keeps `dist/*.whl`: a fresh checkout builds one wheel, and two would fail the script's argument parsing loudly; local runs name the exact file (Global Constraints).
- **C13. The version is bumped with `uv lock`, and its timestamp line is accepted.** After setting 0.2.0 in `pyproject.toml`, `uv lock` changes exactly two lines of `uv.lock`: the `clocks` version and the `[options]` `exclude-newer` timestamp, which uv recomputes from the project's seven-day window; no package changes (measured, K2). Hand-editing the version line alone also passes `uv lock --check` (K2), but it is not how the lock is meant to be written. Cost: one timestamp of churn.
- **C14. Prose guards pin whole forms** (pull request 3's lesson: rule lists drew a new spelling each round). The command block of README and Getting Started is one generated exact text in both files; the upgrade note is one exact text; on the Architecture page, the Visualization subgraph is one exact text and the edges to or from the changed nodes (the subgraph's, `replay` and `demos`) one exact set, and every node label must name a module that exists; the rest of the diagram (layout and unrelated dependencies) is not pinned, by the coordinator's ruling on round 1; the README's `src/clocks/` listing is one exact tuple of entries, each of which must exist.
- **C15. Directory creation is tested in two places.** `write_outputs` creates its directory; one test of the writer and one of the default command (`output/`) start from a missing one, and every other test writes into a directory that exists (the test's own `tmp_path`, a `given` directory it made, or `--output-dir .` with the working directory there). Otherwise a mutant that skips the `mkdir` fails dozens of unrelated tests and the mutation table cannot be exact.
- **C16. Deferred items from pull request 3's reviews** (`gh pr view 29 --comments`; the implementation ledger). Taken, because their files are edited here: `encode` trusting an encoder that exits 0 early, and its cleanup masking the original error (Task 1); the Architecture page and the README module list naming `_animate` and `_panels` (Task 5); the README linking the site twice in its demos section (Task 3, which rewrites that section). Already done on `main` (read): `tests/test_video.py`'s exit-status assertion pins `"status 3"`; the `check_code` docstring Minor of round 9 was corrected in `ca132e1`. Left: the two Python 3.13 symlink-loop cases in `_site_media._refuse_a_linked_media` (the file is not otherwise edited here, the paths are contrived, pull request 3 simplified that guard after three rounds of loop cases, and no 3.13 interpreter was measured: a follow-up of its own); `wait()` without a timeout on `_run_encoder`'s error path (a timeout policy and a test that hangs, which D47 forbids as a mutant); `format` shadowing a builtin in `_style._save_canvas` (not edited here); `reproducibility.qmd` line 91 not re-wrapped (a pinned sentence in a paragraph this plan does not touch; re-wrapping changes no text); the "exactly once" scan reading only `.md` and `.qmd` (unrelated to the commands); and the per-task test-only Minors in files this plan does not edit (`tests/test_clips.py`, `tests/test_check_site_figures.py`, `tests/test_site_media.py`, `tests/test_replays.py`), triaged there as able to wait.
- **C17. The demos import the private modules they use, never the facade.** `clocks.viz` is for library users; package code imports `clocks._clips` and the rest directly, as `_site_media` does. A test keeps the demos off `clocks.viz`, `_animate`, `_panels` and `_panels3d` from Task 3 on, so Task 5's deletions cannot break them. The test resolves relative imports (`from .. import viz`) against `clocks._demos` (Codex, round 1).

## Measurements

Run on 2026-11-26 in this worktree at `6e0527c` (the merge of pull request 3), macOS arm64, Python 3.14.0, Matplotlib 3.10.8, NumPy 2.4.2, uv 0.9.21, FFmpeg 9.0 from `imageio-ffmpeg`. The scratch scripts are not committed; each line says what it ran.

- **K1. Abbreviation.** `argparse.ArgumentParser()` with only `--output-dir`: `parse_args(["--output", "x"])` gives `Namespace(output_dir='x')`. With `allow_abbrev=False` the same call exits with status 2, "unrecognized arguments: --output x". The seven parsers are built today with `argparse.ArgumentParser(description=__doc__)`, the default (read).
- **K2. The version and the lock.** `pyproject.toml` and `uv.lock` (the entry `name = "clocks"`, `version = "0.1.0"`, `source = { editable = "." }`) both say 0.1.0; CI installs with `uv sync --locked`. In a copy of the tree (`git archive HEAD | tar -x -C <copy>`), setting `version = "0.2.0"` and running `uv lock` printed "Updated clocks v0.1.0 -> v0.2.0" and changed two lines of `uv.lock`: that version, and `exclude-newer = "2026-11-17T21:46:28.880603Z"` to the date seven days before the run; no package entry changed. With the original lock, `uv lock --check` fails ("needs to be updated"); with only the version line edited by hand it passes.
- **K3. Who uses the old names.** `git grep -nE "clocks\.viz|_animate|_panels|animate_|create_(echolocation|inference)_dashboard|plot_(centered_rates|clock_rates|clock_setup|mass_histogram|particle_cloud|scene_3d)" -- . ':!docs'`: the removed names are imported by the six clip demo modules, `src/clocks/viz.py` and `tests/test_viz.py` only; `_animate.py` imports `_panels` and `_panels3d`, and nothing else imports any of the three. Prose naming them: `README.md` lines 106 and 109 and `site/reproduce/architecture.qmd` lines 29 to 31, 62 to 69 and 88 to 90. `tests/test_viz.py` holds 18 classes and 39 tests; its package imports are the removed names and `api`, `config`, `inference`, `noise`, `physics` and `types` at module level, and `clocks._scenarios` and `clocks._animate` inside some tests (read; Gemini, round 5); every test in it exercises removed names. Other tests use `clocks.viz` only for the names that stay (`tests/test_clips.py`, `test_figures.py`, `test_echo_study.py`, `test_site_figure_style.py`) and no test imports `_animate`, `_panels` or `_panels3d`.
- **K4. Today's parser test asserts lower bounds.** `tests/test_cli.py`, `test_demo_parser_preserves_defaults_and_accepts_fast_overrides`: `assert defaults.observations > 1` and `assert defaults.particles > 16` (read). Task 3 ties each default to the demo's `scenario()`.
- **K5. Small runs.** For each demo: `record(n_observations=1, n_particles=16)` 0.03 to 0.13 s; `save_replay` and the PNG inside `style("light")` (`final_frame`, or `density_figure` for `demo_density`) 0.04 to 0.46 s; `render_clip` of `record(n_observations=2, n_particles=16)` 0.3 to 3.7 s (echolocation the longest), 92 frames each, and `probe` reports 92 frames. PNG and frame sizes are 1500 by 900 (`demo_1d`, `demo_2d`), 1000 (`demo_multi_mass`), 1100 (`demo_multi_mass_2d`, `demo_echolocation_3d`), 840 (`demo_model_comparison`) and, for the density PNG, 1080: twice `CLIP_HEIGHTS`, and 5.4 inches at 200 dpi.
- **K6. An encoder that exits 0 early.** A stand-in encoder script that writes five bytes to its last argument and exits 0 without reading its input, given to `encode([(bytes(1500 * 900 * 4), 50)], <dir>/clip.mp4, width=1500, height=900, encoder=<script>)` with `b"old"` already at `clip.mp4`: `encode` returned 1, and `clip.mp4` held the five bytes. The 1 counts a frame handed to the pipe, not one the encoder read. `_feed` treats a broken pipe, on a write or on the close, as the encoder having exited and leaves the verdict to its status (read), and `stdin` is a buffered pipe (`subprocess.PIPE` with the default buffering), so frames written into the buffer can be lost at the close after the generator is exhausted (Codex, round 1; read, not measured).
- **K7. A bare wheel.** `uv build --wheel`, then `uv venv` and `uv pip install` of the wheel into it: `imageio_ffmpeg` is not installed there; with the inherited `PATH`, `find_encoder()` found an `ffmpeg` installed system-wide; run with `PYTHONPATH` and `IMAGEIO_FFMPEG_EXE` unset and `PATH` the environment's scripts directory alone, it raised `EncoderNotFound` with "no FFmpeg encoder found: install imageio-ffmpeg (in a checkout, `uv sync` provides it) or put ffmpeg on PATH", and today's `demo-1d` ran to completion under that `PATH` (Matplotlib needs nothing on it).
- **K8. A PNG's corner is its face colour.** For every demo in both themes, the pixel at (0, 0) of the PNG drawn by `final_frame` (or `density_figure`) inside `style(theme)` equals `palette(theme).background` exactly (`#f9f5d7`, `#1d2021`).
- **K9. The suite.** `uv run pytest --collect-only -q -m ''` collects 2,541 tests at `6e0527c`; `tests/test_viz.py` holds 39 of them.
- **K10. Decimals.** `trace_decimals` gives 2 for every position range and 3 for every mass range of the six clip demos (and 4 for the density amplitude, which keeps its own printing), so the commands print 3 and 4.
- **K11. Wrapped arrays.** `f"True rates: {rates}"` for `demo_multi_mass_2d`'s ten rates spans two lines (NumPy's default line width of 75); the echolocation demo has 27 clocks. A printed array is therefore compared as part of the whole output, never as one line (Task 3).

Not measured: the new tests and the mutation table (derived from the code as specified, and checked against the collected IDs in Task 7); the runner's `PATH` (whether `ubuntu-latest` carries an `ffmpeg` is believed, not measured, and is why the wheel check empties `PATH`); the runner's uv accepting the recomputed `exclude-newer` line (believed: it accepted this form at `ca132e1`); the site render at this head (Tasks 3 and 5 run it).

---

### Task 1: `encode` refuses a clip after a broken pipe

**Files:**
- Modify: `src/clocks/_video.py` (`_feed`, `_run_encoder`, `encode`, the `EncoderError` docstring); `src/clocks/_clips.py` (the `render_clip` docstring, which returns `encode`'s count)
- Test: `tests/test_video.py`

**Interfaces:** the seam `_run_encoder(command, frames) -> tuple[int, str]` keeps its signature, so every stand-in in `tests/test_video.py` stands. `_feed(pipe, frames) -> bool` now returns whether every frame was written and the pipe closed without breaking; today it swallows a `BrokenPipeError` on a write *and* on the close (read: `_video.py` lines 292 to 306), and `stdin` is a buffered pipe, so the last frames can be lost at the close after the generator is exhausted (Codex, round 1). `_run_encoder` raises `EncoderError` when the encoder exits 0 although its input broke; a non-zero status is returned as today, so `encode` reports it as today. `encode` is otherwise unchanged in behaviour, except that if removing its temporary file fails while an error propagates, the error propagates with a note naming the file, as `clocks._replay._write_archive` does. `encode`'s count is then the number of frames sent with no broken pipe seen, on a write or on the close; it does not prove the encoder read them (a child that exits 0 without reading can still let every write and the close succeed, Codex round 2, measured there with an operating-system pipe). Whether the file holds every frame is not `encode`'s claim; the commands check it with `probe` before publishing (Task 2), as the site's rendered check does for the site's clips.

- [ ] **Step 1: Write the failing tests** in `tests/test_video.py`, beside the encoder tests (they use the module's `_process_class`, `_small_clip` and `_names`):

```python
@pytest.mark.parametrize(
    ("break_write", "break_close"),
    [pytest.param(True, False, id="write_breaks"), pytest.param(False, True, id="close_breaks")],
)
def test_an_encoder_that_stops_reading_early_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, break_write: bool, break_close: bool
) -> None:
    """Only this catches a partial clip moved into place by an encoder that
    exited 0 after its input broke, on a write or only on the final close."""
    (tmp_path / "clip.mp4").write_bytes(b"old")
    calls: list[str] = []
    base = _process_class(
        calls, status=0, break_write=break_write, break_close=break_close
    )

    class PartialWriter(base):
        def __init__(self, command, *args, **kwargs) -> None:
            super().__init__(command, *args, **kwargs)
            Path(command[-1]).write_bytes(b"partial")

    monkeypatch.setattr(clocks._video.subprocess, "Popen", PartialWriter)

    with pytest.raises(EncoderError) as raised:
        encode(_small_clip(), tmp_path / "clip.mp4", width=64, height=48, encoder="e")

    message = str(raised.value)
    assert "status 0" in message
    assert "before reading all of its input" in message
    assert "clip.mp4" in message
    assert _names(tmp_path) == ["clip.mp4"]
    assert (tmp_path / "clip.mp4").read_bytes() == b"old"


def test_a_failed_cleanup_keeps_the_encoders_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only this catches the cleanup's own error replacing the encoder's."""
    made: list[Path] = []

    def stand_in(command, frames) -> tuple[int, str]:
        temporary = Path(command[-1])
        temporary.mkdir()  # removing a directory with unlink() raises OSError
        made.append(temporary)
        for _ in frames:
            pass
        return 3, "boom"

    monkeypatch.setattr(clocks._video, "_run_encoder", stand_in)

    with pytest.raises(EncoderError, match="status 3") as raised:
        encode(_small_clip(), tmp_path / "clip.mp4", width=64, height=48, encoder="e")

    assert any(made[0].name in note for note in getattr(raised.value, "__notes__", []))
```

  `test_run_encoder_reaps_the_process` stands: its broken-pipe cases exit with status 1, which is returned as today.

- [ ] **Step 2: Run** `uv run pytest tests/test_video.py -q -k "stops_reading_early or failed_cleanup"`. Expected: both cases of the first fail with "DID NOT RAISE" (status 0, so `encode` moved the fake's `b"partial"` onto `clip.mp4`); the second fails with the `OSError` of `unlink` on the directory (on macOS `PermissionError`, on Linux `IsADirectoryError`) in place of the `EncoderError`. Measured while planning, with these tests appended to a scratch copy of `tests/test_video.py`: on `main` the three cases fail exactly so (macOS, `PermissionError`); with Step 3 applied to a scratch copy of `src/`, the whole file passes, `test_run_encoder_reaps_the_process` included.

- [ ] **Step 3: Implement.**

```python
def _feed(pipe: IO[bytes], frames: Iterable[bytes]) -> bool:
    """Write each frame to ``pipe`` and close it; return whether every write and
    the close succeeded with no broken pipe seen.

    ``True`` does not show that the encoder read the frames: a process that
    exits without reading can still let every write and the close succeed.

    A broken pipe, on a write or on the close that flushes the buffer, means
    the encoder stopped reading: writing stops and ``False`` is returned. Only
    a write or the close can break the pipe here: an exception raised while
    taking the next frame, a ``BrokenPipeError`` included, is the caller's.
    """
    complete = True
    for frame in frames:
        try:
            pipe.write(frame)
        except BrokenPipeError:
            complete = False
            break  # the encoder has exited; its status says why
    try:
        pipe.close()
    except BrokenPipeError:
        complete = False  # buffered frames were lost at the flush
    return complete
```

  In `_run_encoder`, `complete = _feed(process.stdin, frames)` in place of the bare call; after the status and the standard error are read, and before returning:

```python
    if status == 0 and not complete:
        raise EncoderError(
            f"{command[0]} exited with status 0 before reading all of its input, "
            f"writing {command[-1]}: {_tail(text)}"
        )
    return status, text
```

  (its docstring gains: "An encoder that exits 0 after a broken pipe, on a write or on the close, raises ``EncoderError``; an encoder that exits 0 having let every write succeed is trusted here, and only a decode of the file can say whether it holds every frame.") In `encode`, the cleanup becomes:

```python
    except BaseException as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            error.add_note(f"{temporary} was left behind: {cleanup_error}")
        raise
```

  The docstring's sentence on the temporary name becomes: "The clip is written beside ``path`` under a temporary name and moved into place only when the encoder exits 0 and no write to it, nor the close of its input, failed with a broken pipe; on any failure ``path`` is left as it was." Its last sentence, "Returns the number of video frames written.", becomes "Returns the number of frames sent with no broken pipe seen, which is not a count of the frames in the file." `EncoderError`'s docstring becomes "The encoder exited with a non-zero status, or exited 0 after a write to it, or the close of its input, broke the pipe." In `render_clip`'s docstring (`src/clocks/_clips.py`), "Returns the number of video frames written." becomes "Returns ``encode``'s count: the frames sent with no broken pipe seen. Only a decode of the file (``clocks._video.probe``) counts the frames it holds."

- [ ] **Step 4: Tests pass; full gate.**

- [ ] **Step 5: Commit and push** as `Refuse a clip whose encoder exited 0 after a broken pipe`. The body cites K6 and the buffered close, and says the cleanup now keeps the original error.

---

### Task 2: `write_outputs`: stage every output, then publish

**Files:**
- Modify: `src/clocks/_demos/_common.py`
- Create: `tests/test_demo_outputs.py`

**Interfaces:**
- Consumes: `save_replay(path, replay)` and `Replay` from `clocks._replay`; `style(theme)`, `save_png(fig, path_or_file)` and `Palette` from `clocks._style`; `render_clip(replay, path, *, theme="light", encoder=None) -> int` and `BUILDERS` (the six clip demos' names, as keys) from `clocks._clips`; `probe(path, encoder=None) -> StreamInfo` (its `frames` is the count the decoder decoded, not one read from the container) and `EncoderError` from `clocks._video`; `matplotlib.pyplot` (only to close a pyplot-held figure: `density_figure`'s is, `final_frame`'s is not, and `plt.close` on the latter does nothing).
- Produces, in `src/clocks/_demos/_common.py` (the module's `positive_int` and its present `add_common_arguments` stay untouched until Task 3):

```python
STAGED = ".staged-"


def write_outputs(
    replay: Replay,
    directory: Path,
    figure: Callable[[Replay, Palette], Figure],
    *,
    theme: str = "light",
    encoder: str | None = None,
) -> list[Path]:
    """Write ``replay``'s archive and PNG into ``directory``, and its clip when
    ``encoder`` is given; return the paths written, archive, PNG, clip.

    Every output is staged under its final name with ``STAGED`` in front, and
    none is moved into place until all are staged and the clip decodes to every
    frame sent (decision C6 of the commands plan). Nothing is printed until
    every rename is made. Without ``encoder``, an entry at ``<demo>.mp4`` (a
    link, dangling or not, included) is kept, and a line says so.
    """
    demo = replay.meta["demo"]
    directory.mkdir(parents=True, exist_ok=True)
    finals = [directory / f"{demo}.npz", directory / f"{demo}.png"]
    if encoder is not None:
        finals.append(directory / f"{demo}.mp4")
    for final in finals:
        # is_symlink first: is_dir follows a link, and on Python 3.12 can raise
        # PermissionError for a link into a directory the caller cannot enter.
        if not final.is_symlink() and final.is_dir():
            raise IsADirectoryError(
                f"{final} is a directory: move it, or choose another --output-dir"
            )
    staged = [final.with_name(STAGED + final.name) for final in finals]
    try:
        for path in staged:
            path.unlink(missing_ok=True)  # left by an interrupted run
        save_replay(staged[0], replay)
        _stage_png(replay, staged[1], figure, theme)
        if encoder is not None:
            sent = render_clip(replay, staged[2], theme=theme, encoder=encoder)
            _check_clip(staged[2], sent, encoder)
        for path, final in zip(staged, finals, strict=True):
            os.replace(path, final)
    except BaseException as error:
        _remove(staged, error)
        raise
    for final in finals:
        print(f"Saved: {final}")
    clip = directory / f"{demo}.mp4"
    has_clip = demo in BUILDERS  # demo_density has no clip and no --video
    if has_clip and encoder is None and (clip.is_symlink() or clip.exists()):
        print(
            f"Kept: {clip} is from an earlier run and was not updated "
            "(--video writes a new one)"
        )
    return finals


def _stage_png(
    replay: Replay, path: Path, figure: Callable[[Replay, Palette], Figure], theme: str
) -> None:
    """Draw the figure inside ``style(theme)`` into a file created at ``path``."""
    with style(theme) as colors:
        fig = figure(replay, colors)
        try:
            with path.open("xb") as file:
                save_png(fig, file)
        finally:
            plt.close(fig)


def _check_clip(path: Path, sent: int, encoder: str) -> None:
    """Refuse, with ``EncoderError``, a staged clip that does not decode to all
    ``sent`` frames (``probe`` runs the same encoder binary as a decoder)."""
    try:
        decoded = probe(path, encoder).frames
    except ValueError as error:
        raise EncoderError(f"{path} does not decode: {error}") from error
    if decoded != sent:
        raise EncoderError(f"{path} decodes to {decoded} of the {sent} frames sent")


def _remove(paths: list[Path], error: BaseException) -> None:
    """Remove ``paths``; a removal that fails is a note on ``error``."""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            error.add_note(f"{path} was left behind: {cleanup_error}")
```

  `save_replay` and `encode` each move a temporary file of their own onto the staged name with `os.replace`, so neither writes through a link there either. The exclusive `open("xb")` guards the moment between the removal of a leftover and the open; no test reproduces that race, so it has no mutant (Task 7). A directory at a final name would make its rename fail after earlier ones had succeeded, so it is refused before anything is staged; a link to a directory is an entry the rename replaces, like any link. A failure during the renames (which, with that preflight, needs the file system itself to fail) propagates through the same handler, which tries to remove the staged files that are left and records any removal that fails as a note on the error; the outputs renamed before it stay, the limit decision C6 states. Printing comes after the renames, so a closed standard output (`BrokenPipeError` from `print`) cannot stop publication part-way.

- [ ] **Step 1: Write the failing tests** in a new `tests/test_demo_outputs.py`:

```python
"""The demo commands' writer: what it writes, and what it leaves alone."""

import os
from pathlib import Path

import matplotlib
import pytest
from _clip_replays import small
from _figure_artists import png_size
from matplotlib.colors import to_hex
from PIL import Image

import clocks._demos._common as common
from clocks._clips import CLIP_HEIGHTS, final_frame
from clocks._demos._common import STAGED, write_outputs
from clocks._figures import density_figure
from clocks._replay import first_difference, load_replay
from clocks._style import palette
from clocks._video import EncoderError, StreamInfo

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
OLD = {"npz": b"old archive", "png": b"old png", "mp4": b"old clip"}


def _existing_set(directory: Path) -> None:
    for suffix, data in OLD.items():
        (directory / f"demo_1d.{suffix}").write_bytes(data)


def _contents(directory: Path) -> dict[str, bytes | None]:
    paths = {suffix: directory / f"demo_1d.{suffix}" for suffix in OLD}
    return {s: p.read_bytes() if p.exists() else None for s, p in paths.items()}


def _staged_names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.startswith(STAGED))


def _corner(path: Path) -> str:
    with Image.open(path) as image:
        return to_hex([c / 255 for c in image.convert("RGB").getpixel((0, 0))])


def _clip_stand_in(log: list, data: bytes, error: BaseException | None = None):
    """Stands in for ``render_clip``, with its defaults."""

    def render(replay, path, *, theme="light", encoder=None):
        log.append((replay, path, theme, encoder))
        path.write_bytes(data)
        if error is not None:
            raise error
        return 92

    return render


def _probe_stand_in(log: list, frames: int = 92, error: Exception | None = None):
    """Stands in for ``probe``: reports ``frames`` decoded, or raises ``error``."""

    def probe(path, encoder=None):
        log.append((path, encoder))
        if error is not None:
            raise error
        return StreamInfo("h264", "yuv420p", 1500, 900, 20.0, frames, True)

    return probe


@pytest.mark.parametrize(
    ("demo", "figure", "height"),
    [
        pytest.param("demo_1d", final_frame, 2 * CLIP_HEIGHTS["demo_1d"], id="demo_1d"),
        pytest.param("demo_density", density_figure, 1080, id="demo_density"),
    ],
)
def test_write_outputs_writes_the_archive_and_png_into_a_new_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], demo, figure, height
) -> None:
    """One of the two tests that start from a missing directory (decision C15),
    and the only one below the commands."""
    replay = small(demo)
    directory = tmp_path / "new" / "out"

    written = write_outputs(replay, directory, figure)

    assert written == [directory / f"{demo}.npz", directory / f"{demo}.png"]
    assert sorted(p.name for p in directory.iterdir()) == [f"{demo}.npz", f"{demo}.png"]
    assert first_difference(replay, load_replay(written[0])) is None
    assert png_size(written[1]) == (1500, height)
    assert capsys.readouterr().out.splitlines() == [f"Saved: {p}" for p in written]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_png_is_the_figure_drawn_for_the_theme(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, theme: str
) -> None:
    """Only this catches the figure handed the light palette while the theme's
    style is in force."""
    replay = small("demo_1d")
    drawn: list = []
    saved: list = []
    real_save = common.save_png

    def figure(given, colors):
        fig = final_frame(given, colors)
        face = to_hex(matplotlib.rcParams["figure.facecolor"])
        drawn.append((given, colors, face, fig))
        return fig

    def save_spy(fig, file) -> None:
        saved.append((fig, file.name))
        real_save(fig, file)

    monkeypatch.setattr(common, "save_png", save_spy)

    write_outputs(replay, tmp_path, figure, theme=theme)

    background = palette(theme).background
    [(given, colors, face, fig)] = drawn
    assert given is replay
    assert colors == palette(theme)
    assert face == background
    assert saved == [(fig, str(tmp_path / f"{STAGED}demo_1d.png"))]
    assert _corner(tmp_path / "demo_1d.png") == background


def test_the_clip_is_drawn_from_the_replay_in_the_theme(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only this catches the clip drawn in the default theme whatever was asked."""
    replay = small("demo_1d")
    log: list = []
    probed: list = []
    monkeypatch.setattr(common, "render_clip", _clip_stand_in(log, b"new clip"))
    monkeypatch.setattr(common, "probe", _probe_stand_in(probed))

    written = write_outputs(
        replay, tmp_path, final_frame, theme="dark", encoder="the-encoder"
    )

    [(given, path, theme, encoder)] = log
    assert given is replay
    assert path == tmp_path / f"{STAGED}demo_1d.mp4"
    assert (theme, encoder) == ("dark", "the-encoder")
    assert probed == [(tmp_path / f"{STAGED}demo_1d.mp4", "the-encoder")]
    assert written[2] == tmp_path / "demo_1d.mp4"
    assert written[2].read_bytes() == b"new clip"
    assert _staged_names(tmp_path) == []


def test_a_failed_clip_leaves_every_existing_output_as_it_was(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only this catches the PNG moved into place before the clip, the last
    output staged, is made (Review Focus 2)."""
    _existing_set(tmp_path)
    failing = _clip_stand_in([], b"partial", EncoderError("the encoder failed"))
    monkeypatch.setattr(common, "render_clip", failing)

    with pytest.raises(EncoderError, match="the encoder failed"):
        write_outputs(small("demo_1d"), tmp_path, final_frame, encoder="the-encoder")

    assert _contents(tmp_path) == OLD
    assert _staged_names(tmp_path) == []
    assert "Saved:" not in capsys.readouterr().out


def test_a_failed_png_leaves_every_existing_output_as_it_was(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only this catches a failure other than the encoder's leaving staged files
    behind (here a PNG save that wrote part of its file)."""
    _existing_set(tmp_path)

    def failing_save(fig, file) -> None:
        file.write(b"partial")
        raise OSError("disk")

    monkeypatch.setattr(common, "save_png", failing_save)

    with pytest.raises(OSError, match="disk"):
        write_outputs(small("demo_1d"), tmp_path, final_frame)

    assert _contents(tmp_path) == OLD
    assert _staged_names(tmp_path) == []


@pytest.mark.parametrize("case", ["earlier_clip", "dangling_clip_link", "no_clip"])
def test_a_run_without_a_clip_keeps_an_earlier_one_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """Only this catches an earlier clip, or a dangling link at its name,
    removed without being asked or kept without a word, or a clip announced
    when there is none (Review Focus 3)."""
    clip = tmp_path / "demo_1d.mp4"
    if case == "earlier_clip":
        _existing_set(tmp_path)
    elif case == "dangling_clip_link":
        try:
            os.symlink(tmp_path / "gone.mp4", clip)
        except OSError:
            pytest.skip("this system refuses symbolic links")

    written = write_outputs(small("demo_1d"), tmp_path, final_frame)

    out = capsys.readouterr().out.splitlines()
    kept = [line for line in out if line.startswith("Kept: ")]
    assert written == [tmp_path / "demo_1d.npz", tmp_path / "demo_1d.png"]
    if case != "no_clip":
        assert clip.is_symlink() == (case == "dangling_clip_link")
        if case == "earlier_clip":
            assert clip.read_bytes() == b"old clip"
        assert kept == [
            f"Kept: {clip} is from an earlier run and was not updated "
            "(--video writes a new one)"
        ]
    else:
        assert not clip.exists()
        assert kept == []


@pytest.mark.parametrize(
    "case", ["final_png_link", "final_archive_link", "staged_png_link", "staged_png_file"]
)
def test_no_output_is_written_through_a_link_or_onto_a_leftover(
    tmp_path: Path, case: str
) -> None:
    """Only this catches an output written through a link at its name, or a
    staged file left by an interrupted run stopping the next one (Review Focus 6)."""
    target = tmp_path / "elsewhere"
    target.write_bytes(b"target")
    name = {
        "final_png_link": "demo_1d.png",
        "final_archive_link": "demo_1d.npz",
        "staged_png_link": f"{STAGED}demo_1d.png",
        "staged_png_file": f"{STAGED}demo_1d.png",
    }[case]
    if case == "staged_png_file":
        (tmp_path / name).write_bytes(b"left over")
    else:
        try:
            os.symlink(target, tmp_path / name)
        except OSError:
            pytest.skip("this system refuses symbolic links")

    write_outputs(small("demo_1d"), tmp_path, final_frame)

    assert target.read_bytes() == b"target"
    assert not (tmp_path / "demo_1d.npz").is_symlink()
    assert not (tmp_path / "demo_1d.png").is_symlink()
    assert load_replay(tmp_path / "demo_1d.npz").meta["demo"] == "demo_1d"
    assert (tmp_path / "demo_1d.png").read_bytes()[:8] == PNG_SIGNATURE
    assert _staged_names(tmp_path) == []


@pytest.mark.parametrize("case", ["short", "undecodable"])
def test_a_clip_that_does_not_decode_to_every_frame_is_not_published(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    """Only this catches a clip published without its decoded frames being
    counted against the frames sent (Review Focus 4)."""
    _existing_set(tmp_path)
    monkeypatch.setattr(common, "render_clip", _clip_stand_in([], b"short clip"))
    probe = (
        _probe_stand_in([], frames=91)
        if case == "short"
        else _probe_stand_in([], error=ValueError("the decoder failed"))
    )
    monkeypatch.setattr(common, "probe", probe)
    expected = "91 of the 92 frames" if case == "short" else "does not decode"

    with pytest.raises(EncoderError, match=expected):
        write_outputs(small("demo_1d"), tmp_path, final_frame, encoder="the-encoder")

    assert _contents(tmp_path) == OLD
    assert _staged_names(tmp_path) == []


def test_a_directory_at_an_output_name_stops_the_run_before_anything_changes(
    tmp_path: Path,
) -> None:
    """Only this catches a directory collision found by a rename part-way
    through publication, after the archive had been replaced."""
    _existing_set(tmp_path)
    (tmp_path / "demo_1d.png").unlink()
    (tmp_path / "demo_1d.png").mkdir()
    (tmp_path / "demo_1d.png" / "kept").write_bytes(b"inside")

    with pytest.raises(IsADirectoryError, match="is a directory"):
        write_outputs(small("demo_1d"), tmp_path, final_frame)

    assert (tmp_path / "demo_1d.npz").read_bytes() == OLD["npz"]
    assert (tmp_path / "demo_1d.mp4").read_bytes() == OLD["mp4"]
    assert (tmp_path / "demo_1d.png" / "kept").read_bytes() == b"inside"
    assert _staged_names(tmp_path) == []


def test_a_failed_rename_removes_the_staged_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only this catches a rename failure leaving staged files behind; it pins
    decision C6's stated limit: the archive renamed before it stays."""
    _existing_set(tmp_path)
    real_replace = os.replace

    def replace(source, destination) -> None:
        if Path(destination).name == "demo_1d.png":
            raise OSError("rename")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", replace)

    with pytest.raises(OSError, match="rename"):
        write_outputs(small("demo_1d"), tmp_path, final_frame)

    assert load_replay(tmp_path / "demo_1d.npz").meta["demo"] == "demo_1d"
    assert (tmp_path / "demo_1d.png").read_bytes() == OLD["png"]
    assert (tmp_path / "demo_1d.mp4").read_bytes() == OLD["mp4"]
    assert _staged_names(tmp_path) == []
    assert "Saved:" not in capsys.readouterr().out


def test_a_closed_output_stream_does_not_stop_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only this catches a closed standard output stopping publication part-way
    (a `Saved:` line printed between renames)."""
    _existing_set(tmp_path)

    def closed(*args, **kwargs) -> None:
        raise BrokenPipeError("standard output is closed")

    monkeypatch.setattr(common, "print", closed, raising=False)

    with pytest.raises(BrokenPipeError):
        write_outputs(small("demo_1d"), tmp_path, final_frame)

    assert load_replay(tmp_path / "demo_1d.npz").meta["demo"] == "demo_1d"
    assert (tmp_path / "demo_1d.png").read_bytes()[:8] == PNG_SIGNATURE
    assert _staged_names(tmp_path) == []


def test_a_failed_cleanup_keeps_the_error_and_removes_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only this catches a staged file that cannot be removed replacing the
    run's own error (here the clip, staged last, so the others are removed
    first either way)."""
    _existing_set(tmp_path)

    def failing(replay, path, *, theme="light", encoder=None):
        path.mkdir()  # removing a directory with unlink() raises OSError
        raise EncoderError("the encoder failed")

    monkeypatch.setattr(common, "render_clip", failing)

    with pytest.raises(EncoderError, match="the encoder failed") as raised:
        write_outputs(small("demo_1d"), tmp_path, final_frame, encoder="the-encoder")

    staged_clip = f"{STAGED}demo_1d.mp4"
    notes = getattr(raised.value, "__notes__", [])
    assert any(staged_clip in note and "left behind" in note for note in notes)
    assert _staged_names(tmp_path) == [staged_clip]
    assert _contents(tmp_path) == OLD


def test_a_link_to_a_directory_at_an_output_name_is_replaced(tmp_path: Path) -> None:
    """Only this catches the directory preflight following a link, which would
    refuse a link the rename can replace (decision C6)."""
    target = tmp_path / "a_directory"
    target.mkdir()
    (target / "kept").write_bytes(b"inside")
    try:
        os.symlink(target, tmp_path / "demo_1d.png", target_is_directory=True)
    except OSError:
        pytest.skip("this system refuses symbolic links")

    write_outputs(small("demo_1d"), tmp_path, final_frame)

    assert not (tmp_path / "demo_1d.png").is_symlink()
    assert (tmp_path / "demo_1d.png").read_bytes()[:8] == PNG_SIGNATURE
    assert (target / "kept").read_bytes() == b"inside"


def test_the_density_demo_says_nothing_of_a_clip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only this catches the `Kept:` line, which names `--video`, printed for
    the density demo, which has no clip and no `--video`."""
    (tmp_path / "demo_density.mp4").write_bytes(b"not ours")

    write_outputs(small("demo_density"), tmp_path, density_figure)

    assert not any(
        line.startswith("Kept: ") for line in capsys.readouterr().out.splitlines()
    )
    assert (tmp_path / "demo_density.mp4").read_bytes() == b"not ours"
```

  (`test_a_failed_rename_removes_the_staged_files` replaces `os.replace` for the whole process; `save_replay`'s own move onto `.staged-demo_1d.npz` passes through it, and only the PNG's publication fails.)

- [ ] **Step 2: Stage the names, then run.** Add to `_common.py` the imports the Interfaces list (`os`, `Callable`, `Figure`, `matplotlib.pyplot as plt`, `Replay`, `save_replay`, `Palette`, `save_png`, `style`, `render_clip`, `BUILDERS`, `probe`, `EncoderError`), `STAGED`, and `write_outputs` with the signature above and the body `return []`. Run `uv run pytest tests/test_demo_outputs.py -q`: every test fails on its assertions (nothing written, no `Saved:` lines, the spies never called, `log` empty, no `EncoderError`, `OSError`, `IsADirectoryError` or `BrokenPipeError` raised), none on an import.

- [ ] **Step 3: Implement** `write_outputs`, `_stage_png`, `_check_clip` and `_remove` as above. `_common.py` now imports `matplotlib.pyplot` and `clocks._clips`; every demo module imports `_common` after `matplotlib.use("Agg")` already (read), and `test_demo_selects_agg_before_importing_pyplot` keeps checking that.

- [ ] **Step 4: Tests pass; full gate.**

- [ ] **Step 5: Commit and push** as `Add the demo writer: stage the archive, PNG and clip, then publish`. The body states decision C6's two limits.

---

### Task 3: The commands: `--output-dir`, `--theme`, `--video`, printing from the replay

**Files:**
- Modify: `src/clocks/_demos/_common.py`, `demo_1d.py`, `demo_2d.py`, `demo_multi_mass.py`, `demo_multi_mass_2d.py`, `demo_model_comparison.py`, `demo_density.py`, `demo_echolocation_3d.py` (all under `src/clocks/_demos/`); `README.md`; `site/reproduce/getting-started.qmd`; `site/reproduce/reproducibility.qmd`; `site/reproduce/architecture.qmd` (one edge)
- Test: `tests/test_cli.py` (rewritten), `tests/test_density_demo.py`, `tests/test_generated_assets.py`, `tests/test_site_figure_style.py` (the stray-video scan, which is the code under test there, as pull request 3 treated that file's guard helpers)

**Interfaces:**
- Consumes: `write_outputs` and `STAGED` (Task 2); `final_frame` from `clocks._clips`; `find_encoder`, `EncoderNotFound`, `EncoderError` from `clocks._video`; `trace_decimals` from `clocks._figures`; each demo module's existing `scenario()` and `record(*, n_observations, n_particles) -> Replay` (read: all seven have both, with the module constants as defaults); `clocks._replay.DEMOS` (the seven names in the order `demo_1d`, `demo_2d`, `demo_multi_mass`, `demo_multi_mass_2d`, `demo_model_comparison`, `demo_density`, `demo_echolocation_3d`).
- Produces, in `src/clocks/_demos/_common.py` (`add_common_arguments` changes signature; `positive_int` stays):

```python
def make_parser(description: str | None) -> argparse.ArgumentParser:
    """A command's parser: long options must be spelt out (decision C4)."""
    return argparse.ArgumentParser(description=description, allow_abbrev=False)


def add_common_arguments(
    parser: argparse.ArgumentParser, *, observations: int, particles: int
) -> None:
    """Add --output-dir, --observations, --particles and --theme."""
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="directory for the replay archive and the PNG (default: output)",
    )
    # --observations and --particles exactly as today, with the defaults passed in
    parser.add_argument(
        "--theme",
        choices=THEMES,
        default="light",
        help="draw the light or the dark rendering (default: light)",
    )


def add_video_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--video",
        action="store_true",
        help="also write the clip, <demo>.mp4, into the output directory; needs "
        "an FFmpeg encoder (imageio-ffmpeg, which `uv sync` installs in a "
        "checkout, or ffmpeg on PATH)",
    )


def run_command(
    parser: argparse.ArgumentParser,
    run: Callable[[argparse.Namespace], None],
    argv: Sequence[str] | None,
) -> int:
    """Parse ``argv``, find the encoder if ``--video`` asks for one, then run."""
    args = parser.parse_args(argv)
    args.encoder = None
    if getattr(args, "video", False):
        try:
            args.encoder = find_encoder()
        except EncoderNotFound as error:
            parser.error(str(error))
    try:
        run(args)
    except EncoderError as error:
        notes = getattr(error, "__notes__", ())
        lines = [f"error: {error}", *(f"note: {note}" for note in notes)]
        parser.exit(1, "".join(f"{parser.prog}: {line}\n" for line in lines))
    return 0


_SUBSCRIPTS = str.maketrans({"₁": "1", "₂": "2"})


def ascii_name(name: str) -> str:
    return name.translate(_SUBSCRIPTS)


def print_truth(replay: Replay) -> None:
    names, values = replay.meta["param_names"], replay.arrays["true_params"]
    pairs = (f"{ascii_name(n)}={v:g}" for n, v in zip(names, values, strict=True))
    print("Truth: " + ", ".join(pairs))
    print(f"True rates: {replay.arrays['true_rates']}")


def print_estimate(replay: Replay) -> None:
    """The last row of a filter replay, one more decimal than the trace headers."""
    arrays, meta = replay.arrays, replay.meta
    sd = np.sqrt(np.diagonal(arrays["cov"][-1]))
    print(f"Final estimate after {meta['n_observations']} observations:")
    rows = zip(meta["param_names"], meta["prior_ranges"], strict=True)
    for j, (name, prior_range) in enumerate(rows):
        d = trace_decimals(prior_range) + 1
        print(
            f"  {ascii_name(name)} = {arrays['mean'][-1, j]:.{d}f} ± {sd[j]:.{d}f}"
            f"  (true: {arrays['true_params'][j]:g})"
        )
    print(f"  ESS = {arrays['ess'][-1]:.0f} / {meta['n_particles']}")
```

  `THEMES` is `clocks._style.THEMES` (`("light", "dark")`).
- Each of the five filter clip demos (`demo_1d`, `demo_2d`, `demo_multi_mass`, `demo_multi_mass_2d`, `demo_echolocation_3d`):

```python
def build_parser() -> argparse.ArgumentParser:
    parser = make_parser(__doc__)
    add_common_arguments(parser, observations=N_OBSERVATIONS, particles=N_PARTICLES)
    add_video_argument(parser)
    return parser


def _run_demo(args: argparse.Namespace) -> None:
    replay = record(n_observations=args.observations, n_particles=args.particles)
    print_truth(replay)
    print_estimate(replay)
    write_outputs(
        replay, args.output_dir, final_frame, theme=args.theme, encoder=args.encoder
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the demo using command-line arguments from *argv*."""
    return run_command(build_parser(), _run_demo, argv)
```

  with `final_frame` imported into the module (`from clocks._clips import final_frame`) and looked up at call time, and the echolocation demo passing `ECHO_N_OBSERVATIONS` and `ECHO_N_PARTICLES`. `OUTPUT_PATH`, the `clocks.viz` import and every import only the old `_run_demo` used go; `scenario`, `record` and the echolocation module's `close_view` and `weight_in_view` stay.
- `demo_model_comparison`: the same parser and `main`; `_run_demo` records once and prints `print_truth(replay)`, then `f"True model: K={meta['true_k']}"`; for k = 20, 40, … up to `n_observations`, `f"After {k} observations:"` and one `f"  K={K}: posterior={p:.4f}"` per candidate from `model_posterior[k]`; then, for the candidate of largest `model_posterior[-1]` (index `best`, `K = model_ks[best]`), `f"MAP model: K={K}"`, `f"Estimate: {arrays[f'final_mean_k{K}']}"`, `f"Std:      {arrays[f'final_std_k{K}']}"` and `f"ESS:      {arrays['model_ess'][-1, best]:.0f} / {meta['n_particles']}"`, with no blank lines; then `write_outputs` as above. The import of `infer` goes: only today's `_run_demo` uses it (`record` uses `simulate` and `build_model_comparison`).
- `demo_density`: `make_parser` and `add_common_arguments`, no `add_video_argument`; its printing unchanged; then `write_outputs(replay, args.output_dir, density_figure, theme=args.theme)`; `main` through `run_command`. Its `OUTPUT_PATH`, `save_png`, `style` and `plt` imports go.

- [ ] **Step 1: Write the failing tests.**

  `tests/test_cli.py` is rewritten whole:

```python
"""Contracts for the packaged demo commands."""

import ast
import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from _clip_replays import committed, edited, small
from _figure_artists import png_size
from matplotlib.colors import to_hex
from PIL import Image

import clocks._demos._common as common
import clocks._video
from clocks._clips import CLIP_HEIGHTS
from clocks._figures import trace_decimals
from clocks._replay import DEMOS, load_replay
from clocks._style import palette
from clocks._video import EncoderError, probe

ROOT = Path(__file__).resolve().parents[1]
CLIP_DEMOS = tuple(name for name in DEMOS if name != "demo_density")
FILTER_CLIP_DEMOS = tuple(n for n in CLIP_DEMOS if n != "demo_model_comparison")
HEIGHTS = {**{n: 2 * CLIP_HEIGHTS[n] for n in CLIP_DEMOS}, "demo_density": 1080}
FAST = ["--observations", "1", "--particles", "16"]


@pytest.fixture(autouse=True)
def _in_tmp_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every test runs in its own empty directory: no run writes into the checkout."""
    monkeypatch.chdir(tmp_path)


def _module(name: str):
    return importlib.import_module(f"clocks._demos.{name}")


def _never(**_: int):
    pytest.fail("the demo was recorded")


def _corner(path: Path) -> str:
    with Image.open(path) as image:
        return to_hex([c / 255 for c in image.convert("RGB").getpixel((0, 0))])


@pytest.mark.parametrize("name", DEMOS)
def test_demo_parser_defaults(name: str) -> None:
    """Only this catches a default run that is not the packaged one."""
    module = _module(name)
    args = module.build_parser().parse_args([])
    packaged = module.scenario()
    assert args.output_dir == Path("output")
    assert args.theme == "light"
    assert (args.observations, args.particles) == (
        packaged.n_observations,
        packaged.n_particles,
    )
    if name in CLIP_DEMOS:
        assert args.video is False


@pytest.mark.parametrize("name", DEMOS)
def test_output_is_no_longer_an_option(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    """Only this catches `--output` still accepted as an alias of `--output-dir`
    (Review Focus 1)."""
    module = _module(name)
    monkeypatch.setattr(module, "record", _never)
    with pytest.raises(SystemExit) as raised:
        module.main(["--output", "x.png", *FAST])
    assert raised.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_options_cannot_be_abbreviated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only this catches a parser that refuses `--output` by name and still
    takes other prefixes: with the test above it shows the cause."""
    module = _module("demo_1d")
    monkeypatch.setattr(module, "record", _never)
    with pytest.raises(SystemExit) as raised:
        module.main(["--obs", "1", "--particles", "16"])
    assert raised.value.code == 2


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("name", DEMOS)
def test_demo_main_writes_its_archive_and_png(
    tmp_path: Path, name: str, theme: str
) -> None:
    """Only this catches a command that writes elsewhere than `--output-dir`,
    records with other counts than it was given, or draws in another theme."""
    given = tmp_path / "given"
    given.mkdir()
    assert _module(name).main(["--output-dir", "given", *FAST, "--theme", theme]) == 0
    assert [p.name for p in tmp_path.iterdir()] == ["given"]
    assert sorted(p.name for p in given.iterdir()) == [f"{name}.npz", f"{name}.png"]
    meta = load_replay(given / f"{name}.npz").meta
    assert (meta["demo"], meta["n_observations"], meta["n_particles"]) == (name, 1, 16)
    assert png_size(given / f"{name}.png") == (1500, HEIGHTS[name])
    assert _corner(given / f"{name}.png") == palette(theme).background


def test_the_default_run_writes_into_output(tmp_path: Path) -> None:
    """One of the two tests that start from a missing directory (decision C15):
    only this catches a default output directory other than `output/`."""
    assert _module("demo_1d").main(FAST) == 0
    assert [p.name for p in tmp_path.iterdir()] == ["output"]
    names = sorted(p.name for p in (tmp_path / "output").iterdir())
    assert names == ["demo_1d.npz", "demo_1d.png"]


@pytest.mark.parametrize("name", CLIP_DEMOS)
def test_png_is_the_final_frame(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Only this catches a clip demo's PNG drawn by another figure function, or
    from another replay than the one archived."""
    module = _module(name)
    archived: list = []
    drawn: list = []
    saved: list = []
    real_archive, real_frame = common.save_replay, module.final_frame
    real_save = common.save_png

    def archive_spy(path, replay) -> None:
        archived.append(replay)
        real_archive(path, replay)

    def frame_spy(replay, colors):
        fig = real_frame(replay, colors)
        drawn.append((replay, fig))
        return fig

    def save_spy(fig, file) -> None:
        saved.append(fig)
        real_save(fig, file)

    monkeypatch.setattr(common, "save_replay", archive_spy)
    monkeypatch.setattr(module, "final_frame", frame_spy)
    monkeypatch.setattr(common, "save_png", save_spy)

    module.main(["--output-dir", ".", *FAST])

    [replay] = archived
    [(drawn_replay, fig)] = drawn
    assert drawn_replay is replay
    assert saved == [fig]


@pytest.mark.parametrize("name", CLIP_DEMOS)
def test_video_writes_the_clip(tmp_path: Path, name: str) -> None:
    """Only this catches a demo whose `--video` writes no clip. It needs the
    encoder and fails, not skips, without one (pull request 3's D11)."""
    args = ["--output-dir", ".", "--video", "--observations", "2", "--particles", "16"]
    assert _module(name).main(args) == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == sorted(f"{name}.{suffix}" for suffix in ("mp4", "npz", "png"))
    info = probe(tmp_path / f"{name}.mp4")
    assert (info.codec, info.pix_fmt) == ("h264", "yuv420p")
    assert (info.frames, info.width, info.height) == (92, 1500, HEIGHTS[name])


def test_density_has_no_video_option(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only this catches `--video` offered by the density demo, which has no clip."""
    module = _module("demo_density")
    monkeypatch.setattr(module, "record", _never)
    with pytest.raises(SystemExit) as raised:
        module.main(["--video", *FAST])
    assert raised.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", CLIP_DEMOS)
def test_video_without_an_encoder_stops_before_the_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
) -> None:
    """Only this catches a missing encoder reaching the user as a traceback
    rather than the usage error and the library's instructions (decision C5)."""
    module = _module(name)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr(clocks._video.shutil, "which", lambda _: None)
    monkeypatch.setattr(module, "record", _never)
    with pytest.raises(SystemExit) as raised:
        module.main(["--output-dir", "out", "--video", *FAST])
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert "imageio-ffmpeg" in err
    assert "ffmpeg on PATH" in err
    assert list(tmp_path.iterdir()) == []


def test_a_failed_encode_ends_with_a_message_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only this catches an encoder failure reaching the user as a traceback, or
    without the notes that say what was left behind."""

    def failing(replay, path, *, theme="light", encoder=None):
        error = EncoderError("the encoder failed")
        error.add_note("a-file was left behind: busy")
        raise error

    monkeypatch.setattr(common, "find_encoder", lambda: "the-encoder")
    monkeypatch.setattr(common, "render_clip", failing)
    with pytest.raises(SystemExit) as raised:
        _module("demo_1d").main(["--output-dir", ".", "--video", *FAST])
    assert raised.value.code == 1
    lines = capsys.readouterr().err.splitlines()
    assert lines[-2].endswith(": error: the encoder failed")
    assert lines[-1].endswith(": note: a-file was left behind: busy")


def _plain(name: str) -> str:
    return name.replace("₁", "1").replace("₂", "2")


@pytest.mark.parametrize("name", FILTER_CLIP_DEMOS)
def test_command_prints_the_recorded_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    """Only this catches numbers printed from the module's constants or the
    command's arguments rather than the replay, or with the headers' decimals."""
    base = small(name)
    width = float(np.diff(base.meta["prior_ranges"][0])[0])
    rates = base.arrays["true_rates"].copy()
    rates[0] = 0.5
    truth = base.arrays["true_params"].copy()
    truth[0] += 0.125 * width
    mean = base.arrays["mean"].copy()
    mean[-1, 0] += 0.25 * width
    stub = edited(base, true_rates=rates, true_params=truth, mean=mean)
    calls: list[dict] = []

    def record(**keywords: int):
        calls.append(keywords)
        return stub

    module = _module(name)
    monkeypatch.setattr(module, "record", record)

    module.main(["--output-dir", ".", "--observations", "7", "--particles", "23"])

    assert calls == [{"n_observations": 7, "n_particles": 23}]
    out = capsys.readouterr().out
    a, meta = stub.arrays, stub.meta
    names = [_plain(n) for n in meta["param_names"]]
    pairs = ", ".join(f"{n}={v:g}" for n, v in zip(names, a["true_params"]))
    # NumPy wraps a long array over several lines (10 rates for the two-mass
    # 2D demo, 27 for echolocation), so the prefix is compared before splitting.
    assert out.startswith(f"Truth: {pairs}\nTrue rates: {a['true_rates']}\n")
    lines = out.splitlines()
    at = lines.index(f"Final estimate after {meta['n_observations']} observations:")
    sd = np.sqrt(np.diagonal(a["cov"][-1]))
    expected = []
    for j, (n, prior_range) in enumerate(zip(names, meta["prior_ranges"])):
        d = trace_decimals(prior_range) + 1
        expected.append(
            f"  {n} = {a['mean'][-1, j]:.{d}f} ± {sd[j]:.{d}f}"
            f"  (true: {a['true_params'][j]:g})"
        )
    expected.append(f"  ESS = {a['ess'][-1]:.0f} / {meta['n_particles']}")
    assert lines[at + 1 : at + 1 + len(expected)] == expected


def test_model_comparison_command_prints_from_the_replay_and_records_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only this catches the comparison recorded a second time for its printout,
    or its truth printed from the module's constants rather than the replay."""
    base = committed("demo_model_comparison")
    rates = base.arrays["true_rates"].copy()
    rates[0] = 0.5
    truth = base.arrays["true_params"].copy()
    truth[0] += 0.5
    stub = edited(base, true_rates=rates, true_params=truth)
    calls: list[dict] = []

    def record(**keywords: int):
        calls.append(keywords)
        return stub

    module = _module("demo_model_comparison")
    monkeypatch.setattr(module, "record", record)

    module.main(["--output-dir", ".", *FAST])

    assert len(calls) == 1
    out = capsys.readouterr().out
    a, meta = stub.arrays, stub.meta
    names = [_plain(n) for n in meta["param_names"]]
    pairs = ", ".join(f"{n}={v:g}" for n, v in zip(names, a["true_params"]))
    assert out.startswith(
        f"Truth: {pairs}\nTrue rates: {a['true_rates']}\n"
        f"True model: K={meta['true_k']}\n"
    )
    ks = [int(k) for k in a["model_ks"]]
    expected = [f"True model: K={meta['true_k']}"]
    for k in range(20, meta["n_observations"] + 1, 20):
        expected.append(f"After {k} observations:")
        posterior = a["model_posterior"][k]
        expected += [f"  K={K}: posterior={p:.4f}" for K, p in zip(ks, posterior)]
    best = int(np.argmax(a["model_posterior"][-1]))
    K = ks[best]
    expected += [
        f"MAP model: K={K}",
        f"Estimate: {a[f'final_mean_k{K}']}",
        f"Std:      {a[f'final_std_k{K}']}",
        f"ESS:      {a['model_ess'][-1, best]:.0f} / {meta['n_particles']}",
    ]
    assert "\n" + "\n".join(expected) + "\n" in out  # whole lines, wrapped or not


_OLD_DRAWING = {"clocks.viz", "clocks._animate", "clocks._panels", "clocks._panels3d"}
_DEMOS_PACKAGE = ("clocks", "_demos")


def _absolute(node: ast.ImportFrom) -> str:
    """The module a `from` import in `clocks._demos` names, relative or not."""
    if not node.level:
        return node.module or ""
    base = _DEMOS_PACKAGE[: len(_DEMOS_PACKAGE) - node.level + 1]
    return ".".join([*base, *([node.module] if node.module else [])])


def test_no_demo_module_imports_the_facade_or_the_old_drawing_code() -> None:
    """Only this catches a demo importing `clocks.viz` or the GIF code, which
    Task 5 of the commands plan deletes (decision C17)."""
    found: dict[str, list[str]] = {}
    for path in sorted((ROOT / "src" / "clocks" / "_demos").glob("*.py")):
        names: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = _absolute(node)
                names.add(module)
                names.update(f"{module}.{alias.name}" for alias in node.names)
        if names & _OLD_DRAWING:
            found[path.name] = sorted(names & _OLD_DRAWING)
    assert found == {}
```

  followed by today's four structural tests, kept with their bodies and parametrized by `name` over `DEMOS` (the executable is `name.replace("_", "-")`): `test_demo_help_exits_without_running`, `test_project_scripts_point_directly_to_packaged_demo_modules` (reading `ROOT / "pyproject.toml"`), `test_demo_selects_agg_before_importing_pyplot` and `test_source_demo_is_only_a_thin_wrapper` (reading `ROOT / "scripts" / f"{name}.py"`). The two tests they replace, `test_demo_parser_preserves_defaults_and_accepts_fast_overrides` and `test_demo_main_executes_with_fast_overrides`, go.

  In `tests/test_density_demo.py`, every test that calls `main` first runs `monkeypatch.chdir(tmp_path)`, so no run can write into the checkout. `test_default_format_prints_the_amplitude_to_four_places`, `test_density_command_prints_and_draws_the_recorded_run`, `test_density_command_prints_progress_from_the_replay`, `test_density_command_saves_the_figure_it_drew` and `test_a_failed_save_still_closes_the_figure` each pass `"--output-dir", str(tmp_path)` exactly where they passed `--output` and its path, keeping their other flags (decision C15: none of them may fall back to the missing default `output/`); a test that has no `monkeypatch` parameter yet (`test_density_command_prints_and_draws_the_recorded_run`) gains one, and the second reads `tmp_path / "demo_density.png"`. `test_density_command_saves_the_figure_it_drew` keeps its figure spy and its assertions on the replay's counts, the light palette and the face colour, and its save spy moves to `clocks._demos._common.save_png` (taking `(fig, file)`): it asserts `saved[0][0] is fig` and drops the path (the staged name is `test_the_png_is_the_figure_drawn_for_the_theme`'s), so a same-size, same-colour substitute figure still fails it. `test_a_failed_save_still_closes_the_figure` patches `clocks._demos._common.save_png` the same way. Deleted: `test_density_command_writes_png_bytes_at_the_path_given` and `test_density_command_writes_only_its_png` (decision C8; what the density command writes is `test_demo_main_writes_its_archive_and_png[demo_density-*]`'s). Remove the imports ruff then reports unused.

  In `tests/test_generated_assets.py` (it imports `DEMOS` from `clocks._replay` beside `load_replay`):

```python
FENCE = "`" * 3


def _command_block() -> str:
    """The command list README and Getting Started both show (decision C14)."""
    lines = [
        f"uv run {name.replace('_', '-'):<25}"
        f"# → output/{name}.npz, output/{name}.png"
        for name in DEMOS
    ]
    lines.append(f"uv run {'demo-2d --video':<25}# also → output/demo_2d.mp4")
    return (
        "Seven demos. Each writes a replay archive and a PNG; the six animated\n"
        "ones also write an MP4 with `--video`:\n\n"
        f"{FENCE}bash\n" + "\n".join(lines) + f"\n{FENCE}\n"
    )


@pytest.mark.parametrize(
    "relative", ["README.md", "site/reproduce/getting-started.qmd"]
)
def test_the_command_block_is_the_commands(relative: str) -> None:
    """Only this catches a command list that still promises GIFs, offers
    `--video` to the density demo, or names a file the commands do not write."""
    text = (ROOT / relative).read_text(encoding="utf-8")
    assert text.count(_command_block()) == 1
    assert ".gif" not in text
```

  In `tests/test_site_figure_style.py`, the Getting Started block's `output/demo_2d.mp4` would trip `test_no_page_has_unpinned_video_markup[site/reproduce/getting-started.qmd]`: `_VIDEO_MARKUP` holds `".mp4"` and `_replay_embed_violations` flags it on any line outside a pinned embed (read; Gemini, round 5). A command's own output path is not video markup, so the scan drops it first, by the prefix rule `_retired_asset_mentions` uses for `output/` paths:

```python
# A command's own output clip, `output/<name>.mp4` standing alone (the rule
# `_retired_asset_mentions` uses for `output/`), is a path, not an embed.
_COMMAND_CLIP = re.compile(r"(?<![\w./-])output/[\w.-]+\.mp4")
```

  and in `_replay_embed_violations` the stray-markup comprehension tests `_COMMAND_CLIP.sub("", line).lower()` in place of `line.lower()`. `test_replay_embed_violations` gains two cases:

```python
        pytest.param(
            PAGE_TEXT + "uv run demo-2d --video   # also → output/demo_2d.mp4\n",
            (_EMBED_2D,),
            [],
            id="a_command_output_clip",
        ),
        pytest.param(
            PAGE_TEXT + "[the clip](assets/output/demo_2d.mp4)\n",
            (_EMBED_2D,),
            ["unpinned video markup: line 14"],
            id="an_output_clip_under_another_directory",
        ),
```

  (`a_link_to_a_clip`, `../media/demo_2d.light.mp4`, stays refused.)

  `test_the_commands_are_described_as_they_are` becomes:

```python
def test_the_commands_are_described_as_they_are() -> None:
    """Only this catches the Reproducibility page describing the commands as they
    were before pull request 4."""
    page = " ".join(_reproducibility_page().split())
    assert (
        "Each `uv run demo-*` command records its configured scenario again and "
        "writes its own archive and a PNG drawn from it: the final frame for the "
        "six animated demos, the density figure for `demo-density`. With "
        "`--video`, the six also write the clip, drawn as the site's are."
    ) in page
    assert "still write GIFs" not in page
    assert "records the same configured scenario and draws its result" not in page
```

  `test_readme_describes_the_site_and_the_commands_as_they_are` becomes:

```python
def test_readme_describes_the_site_and_the_commands_as_they_are() -> None:
    """Only this catches the README's paragraphs on the commands and on what is
    committed losing a sentence, or linking the site twice."""
    readme = (ROOT / "README.md").read_text()
    flat = " ".join(readme.split())
    for sentence in (
        "Each command records its run, prints the truth and its final estimate "
        "from the recording, and writes a replay archive and a PNG: the final "
        "frame of the animation for six of them, the density figure for "
        "`demo-density`.",
        "The six animated demos also take `--video`, which writes the animation "
        "as an MP4 and needs an FFmpeg encoder (`uv sync` provides one in a "
        "checkout).",
        "`--theme dark` draws the dark version, and `--output-dir` chooses the "
        "directory (`output/` by default).",
        "A run without `--video` leaves an MP4 from an earlier run where it is "
        "and says that it was not updated.",
        "The site animates six of the demos and draws the density figure, all "
        "from replay data",
        "The site's clips and posters are build products and are not committed.",
        "What is committed is replay data, `assets/replays/<demo>.npz`: what each "
        "packaged run produced, in numbers, tied to the code by tests of its "
        "inputs and of bounded summaries.",
    ):
        assert sentence in flat
    section = readme.split("## Run the demos", 1)[1].split("## Run tests", 1)[0]
    assert section.count("](https://jbwhit.github.io/clocks/)") == 1
    for stale in ("still write GIFs", "assets/demo_2d.gif", "commits no rendered media"):
        assert stale not in flat
```

- [ ] **Step 2: Run against today's code, then stage the mechanical part, then run again.** First run `uv run pytest tests/test_cli.py tests/test_density_demo.py tests/test_generated_assets.py tests/test_site_figure_style.py -q` as the tree stands (after Task 2). Expected: `test_demo_parser_defaults` fails (no `output_dir`); `test_output_is_no_longer_an_option` fails for every demo, since today's parser takes `--output x.png` and the run starts (for the clip demos, Matplotlib's movie writer is asked for a `.png`, which raises; for `demo_density`, `record` is `_never`), so no `SystemExit` comes; `test_png_is_the_final_frame` and `test_a_failed_encode_ends_with_a_message_not_a_traceback` fail at `monkeypatch.setattr`, with `AttributeError` (the modules have no `final_frame`, and Task 2 gave `_common` no `find_encoder`); the tests that pass `--output-dir` exit with status 2 before their assertions (`test_video_without_an_encoder_stops_before_the_run` then fails on the message); the command-block and both prose tests fail (the files still list GIFs), as does `test_no_demo_module_imports_the_facade_or_the_old_drawing_code`, listing the six clip demos; `a_command_output_clip` fails (line 14 flagged). `test_options_cannot_be_abbreviated` fails too (`--obs` is an unambiguous prefix of `--observations` today, so today's `demo-1d` run starts). `test_density_has_no_video_option` already passes, since today's density parser has no `--video`: it is the one test that is green before and after. Where the run's output differs from this list, the output is what is recorded. Then stage the mechanical part: in `_common.py`, `make_parser`, the new `add_common_arguments`, `add_video_argument` and `run_command` as the Interfaces give them, with the imports they need, and `ascii_name`, `print_truth` and `print_estimate` with bodies that print nothing; in each demo module, `build_parser` and `main` as the Interfaces give them, `final_frame` (or `density_figure`) imported, the import from `clocks.viz` removed, and `_run_demo` replaced by a stub that does nothing (`pass`). Run the same command again. Expected now, each on its assertion: no archive or PNG written (`test_demo_main_writes_its_archive_and_png`, `test_the_default_run_writes_into_output`, `test_video_writes_the_clip`, the density command tests), `record` never called with the given counts (`test_command_prints_the_recorded_run`, the model-comparison test), `final_frame` never called (`test_png_is_the_final_frame`), no `SystemExit(1)` (`test_a_failed_encode_ends_with_a_message_not_a_traceback`); the parser tests and `test_video_without_an_encoder_stops_before_the_run` now pass, since staging them is their implementation.

- [ ] **Step 3: Implement** the `_common.py` functions as above and the seven demo modules as the Interfaces say. Imports in each demo module stay after `matplotlib.use("Agg")`.

  In `README.md`, "## Run the demos" becomes, in this order: the text `_command_block()` returns (the line "Six animated demos and one static figure:" and the old block go); then the paragraph

  > Each command records its run, prints the truth and its final estimate from the recording, and writes a replay archive and a PNG: the final frame of the animation for six of them, the density figure for `demo-density`. The six animated demos also take `--video`, which writes the animation as an MP4 and needs an FFmpeg encoder (`uv sync` provides one in a checkout). `--theme dark` draws the dark version, and `--output-dir` chooses the directory (`output/` by default). A run without `--video` leaves an MP4 from an earlier run where it is and says that it was not updated.

  then the paragraph

  > The site animates six of the demos and draws the density figure, all from replay data, with commentary: [jbwhit.github.io/clocks](https://jbwhit.github.io/clocks/). The site's clips and posters are build products and are not committed. What is committed is replay data, `assets/replays/<demo>.npz`: what each packaged run produced, in numbers, tied to the code by tests of its inputs and of bounded summaries. The echolocation range study behind the site's final page: `scripts/scan_echolocation_range.py`.

  replacing the two paragraphs there today (the sentence "Six of the commands above still write GIFs…" and the paragraph "Most GIF demos animate…", with its second link to the site, go). On the Architecture page, whose solid arrows are imports, the edge `    demos --> viz` becomes `    demos --> figstyle` (the demos now import `clocks._style` through `_common`, and no longer import `clocks.viz`); Task 5 redraws the rest of that diagram. Wrap at about 78 columns, as the file does. In `site/reproduce/getting-started.qmd`, "## Run the demos" holds the same `_command_block()` text in place of its line "Six animated demos plus the density figure:" and its block. On the Reproducibility page, the two sentences "`uv run demo-density` records the same configured scenario and draws its result. The other six `uv run demo-*` commands run the same scenarios again and still write GIFs, drawn by the earlier animation code and not from these archives." become the two the test pins. `site/story/gravitational-echolocation.qmd`'s "`uv run demo-echolocation-3d` regenerates the demo;" stays true and is not touched.

- [ ] **Step 4: Tests pass; full gate; render the site and run both checks** (three pages changed: Getting Started, Reproducibility, Architecture); **then the wheel check** on `dist/clocks-0.1.0-py3-none-any.whl` (every command's `--help` still prints usage).

- [ ] **Step 5: Commit and push** as `Demo commands record, print from the replay and write an archive and a PNG; --video adds the clip`. The body says that `--output` is gone and options can no longer be abbreviated, what each command writes, what happens to an older clip, and that the model-comparison command now records once.

---

### Task 4: The wheel check runs the default command with no encoder, and `--video` without one

**Files:**
- Modify: `scripts/check_wheel_entrypoints.py`
- Create: `tests/test_wheel_check.py`

**Interfaces:**
- Produces, in the script:

```python
_INHERITED_WAYS_IN = frozenset({"PATH", "PYTHONPATH", "IMAGEIO_FFMPEG_EXE"})


def bare_environment(
    scripts_directory: Path, inherited: Mapping[str, str]
) -> dict[str, str]:
    """``inherited`` without ``PYTHONPATH`` or ``IMAGEIO_FFMPEG_EXE``, and with
    ``PATH`` the environment's scripts directory alone, so neither a checkout
    nor an encoder can be reached by inheritance (K7 of the commands plan)."""
    environment = {k: v for k, v in inherited.items() if k not in _INHERITED_WAYS_IN}
    environment["PATH"] = str(scripts_directory)
    return environment
```

  and `_check_the_commands_without_an_encoder(scripts_directory: Path, suffix: str) -> None`, called from `main` after the `--help` loop. In a new temporary directory, with `env=bare_environment(scripts_directory, os.environ)` and `cwd` that directory, it runs `demo-1d --output-dir out --observations 2 --particles 16` (by its full path in the scripts directory) and requires exit status 0 and `out/` holding exactly `demo_1d.npz` and `demo_1d.png`; prints `ok: demo-1d writes its archive and PNG with no encoder`; reads both files' bytes; runs the same command with `--video` and requires exit status 2, `imageio-ffmpeg` in its standard error, and `out/` holding the same two files with the same bytes; prints `ok: demo-1d --video stops and names imageio-ffmpeg when there is no encoder`. Any other outcome raises `RuntimeError` with the command, its status and both streams. If an encoder were reachable, the `--video` run would exit 0 and the check would fail, so the check cannot pass by finding one.

- [ ] **Step 1: Write the failing test** in a new `tests/test_wheel_check.py`:

```python
"""The wheel check's environment. The check itself installs a wheel, which is
CI's job (decision C12 of the commands plan)."""

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "check_wheel_entrypoints", ROOT / "scripts" / "check_wheel_entrypoints.py"
)
assert _SPEC is not None and _SPEC.loader is not None
check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check)


def test_the_bare_environment_reaches_no_checkout_and_no_encoder() -> None:
    """Only this catches the wheel check inheriting a way to the checkout or to
    an encoder: `PATH` beyond the environment's scripts, `PYTHONPATH`, or
    `IMAGEIO_FFMPEG_EXE`."""
    inherited = {
        "PATH": os.pathsep.join(["usr-bin", "opt-bin"]),
        "PYTHONPATH": "src",
        "IMAGEIO_FFMPEG_EXE": "opt-bin/ffmpeg",
        "HOME": "home",
        "LANG": "C",
    }
    before = dict(inherited)

    environment = check.bare_environment(Path("venv", "bin"), inherited)

    assert environment == {
        "PATH": str(Path("venv", "bin")),
        "HOME": "home",
        "LANG": "C",
    }
    assert inherited == before
```

- [ ] **Step 2: Stage the name, then run.** Add `bare_environment` returning `dict(inherited)`. Run `uv run pytest tests/test_wheel_check.py -q`: the test fails on the equality (all five keys returned).

- [ ] **Step 3: Implement** `bare_environment` and `_check_the_commands_without_an_encoder` as above; `_run` is left as it is for the existing steps, and the two new runs use `subprocess.run(..., capture_output=True, text=True, check=False, cwd=..., env=...)` directly, since the second must fail.

- [ ] **Step 4: Tests pass; full gate; the wheel check** on `dist/clocks-0.1.0-py3-none-any.whl`, which must end with the two new `ok:` lines.

- [ ] **Step 5: Commit and push** as `Wheel check: the default command works with no encoder; --video without one stops`. The body cites K7.

---

### Task 5: Remove the GIF animators, the panel functions and their public names

**Files:**
- Delete: `src/clocks/_animate.py`, `src/clocks/_panels.py`, `src/clocks/_panels3d.py`, `tests/test_viz.py`
- Modify: `src/clocks/viz.py`, `README.md` (the "Project structure" listing), `site/reproduce/architecture.qmd`
- Test: `tests/test_api.py`, `tests/test_generated_assets.py`

- [ ] **Step 1: Confirm nothing else uses what is being deleted.** The pattern is built from the three module names and all nineteen removed names, with POSIX character classes for the word boundaries (`git grep -E` does not honour `\b` here: an earlier draft's pattern missed `viz.py`'s own imports, Codex round 1, confirmed while revising):

```bash
git grep -lE '(^|[^[:alnum:]_])(_animate|_panels|_panels3d|animate_(echolocation|inference|inference_2d|inference_multi_1d|inference_multi_2d|model_comparison)|create_(echolocation|inference)_dashboard|plot_(centered_rates|clock_rates|clock_rates_2d|clock_setup|clock_setup_2d|mass_histogram|particle_cloud|particle_cloud_2d|particle_cloud_multi_1d|particle_cloud_multi_2d|scene_3d))([^[:alnum:]_]|$)' -- src scripts site tests README.md ':(glob)docs/**/*.py'
```

  At `6e0527c` it lists `README.md`, `site/reproduce/architecture.qmd`, the six clip demo modules, `src/clocks/_animate.py`, `_panels.py`, `_panels3d.py`, `viz.py` and `tests/test_viz.py` (measured). After Task 3 it must list the same files without the six demo modules and with `tests/test_cli.py`, whose one matching line is Task 3's negative guard, `_OLD_DRAWING = {"clocks.viz", "clocks._animate", "clocks._panels", "clocks._panels3d"}` (Codex, round 2); confirm that with the same pattern and `-n` on `tests/test_cli.py`, which must print that line and no other. If the search lists anything else, or `tests/test_cli.py` matches elsewhere, stop and report.

- [ ] **Step 2: Write the failing tests.** In `tests/test_api.py` (importing `importlib.util` and `clocks.viz`):

```python
ANIMATE = (
    "animate_echolocation", "animate_inference", "animate_inference_2d",
    "animate_inference_multi_1d", "animate_inference_multi_2d",
    "animate_model_comparison",
)  # fmt: skip
PLOT = (
    "plot_centered_rates", "plot_clock_rates", "plot_clock_rates_2d",
    "plot_clock_setup", "plot_clock_setup_2d", "plot_mass_histogram",
    "plot_particle_cloud", "plot_particle_cloud_2d", "plot_particle_cloud_multi_1d",
    "plot_particle_cloud_multi_2d", "plot_scene_3d",
)  # fmt: skip
DASHBOARD = ("create_echolocation_dashboard", "create_inference_dashboard")
VIZ_NAMES = {
    "THEMES", "Palette", "density_figure", "load_replay", "palette",
    "range_study_figure", "render_clip", "show", "style",
}  # fmt: skip


def test_viz_exports_exactly_the_documented_names() -> None:
    """Only this catches a public name added to `clocks.viz` without the upgrade
    note changing."""
    assert sorted(clocks.viz.__all__) == sorted(VIZ_NAMES)
    assert all(hasattr(clocks.viz, name) for name in VIZ_NAMES)


def test_the_removed_names_are_gone() -> None:
    """Only this catches a removed name still importable from `clocks.viz`
    though no longer listed, or an old module still in the package."""
    for name in (*ANIMATE, *PLOT, *DASHBOARD):
        assert not hasattr(clocks.viz, name), name
    for module in ("clocks._animate", "clocks._panels", "clocks._panels3d"):
        assert importlib.util.find_spec(module) is None, module
```

  In `tests/test_generated_assets.py`:

```python
# The parts of the Architecture page this plan changes (decision C14): the
# Visualization subgraph, every edge to or from a node of it, the replay archive
# or the demos, and two sentences. The rest of the diagram is not pinned.
ARCHITECTURE_VIS_SUBGRAPH = "\n".join(
    (
        '    subgraph vis["Visualization"]',
        "        viz",
        '        figures["_figures"]',
        '        clips["_clips, _clip_*"]',
        '        video["_video"]',
        '        sitemedia["_site_media, _sources"]',
        '        figstyle["_style, _fonts/"]',
        "    end",
    )
)
ARCHITECTURE_EDGES = {
    ("viz", "figures"), ("viz", "clips"), ("viz", "echo"), ("viz", "replay"),
    ("viz", "figstyle"), ("figures", "replay"), ("figures", "figstyle"),
    ("clips", "figures"), ("clips", "replay"), ("clips", "video"),
    ("clips", "figstyle"), ("sitemedia", "clips"), ("sitemedia", "replay"),
    ("sitemedia", "video"), ("sitemedia", "figstyle"), ("replay", "api"),
    ("replay", "config"), ("replay", "types"), ("echo", "figstyle"),
    ("demos", "api"), ("demos", "scenario"), ("demos", "replay"),
    ("demos", "figures"), ("demos", "clips"), ("demos", "video"),
    ("demos", "figstyle"),
}  # fmt: skip
_CHANGED_NODES = {
    "viz", "figures", "clips", "video", "sitemedia", "figstyle", "replay", "demos",
}  # fmt: skip
ARCHITECTURE_SENTENCES = (
    "`viz` is the public facade over the private figure and clip modules "
    "(`_figures`, `_echo_study` for the range-study figure, and `_clips` with "
    "its `_clip_*` builders), the replay archive "
    "(`_replay`) and the shared figure style (`_style`, with its bundled fonts); "
    "`_video` encodes and checks the clips, and `_site_media` draws the site's "
    "clips and posters before Quarto renders.",
    "Modules whose names begin with an underscore, including `_support`, "
    "`_scenarios`, `_echo_study`, `_replay`, `_figures`, `_clips` and `_demos`, "
    "are internal; library callers import from `clocks`, and the figure, clip "
    "and style names from `clocks.viz`.",
)
_NOT_MODULES = {"console", "wrappers", "end"}


def _mermaid(page: str) -> str:
    return page.split("```{mermaid}\n", 1)[1].split("\n```", 1)[0]


def _diagram_modules(block: str) -> list[str]:
    """Each module a node of ``block`` names: its label split on ``, ``."""
    names = []
    for line in block.splitlines():
        match = re.fullmatch(r'\s+(\w+)(?:\["([^"]+)"\])?', line)
        if match and match.group(1) not in _NOT_MODULES:
            names += (match.group(2) or match.group(1)).split(", ")
    return names


def _in_package(name: str) -> bool:
    package = ROOT / "src" / "clocks"
    if name.endswith("/"):
        return (package / name.rstrip("/")).is_dir()
    if name.endswith("/*"):
        return (package / name.removesuffix("/*")).is_dir()
    if name.endswith("*"):
        return any(package.glob(f"{name}.py"))
    return (package / f"{name}.py").is_file()


def test_the_architecture_page_maps_the_drawing_modules_as_they_are() -> None:
    """Only this catches the module map's visualization part or the demos'
    imports drawn otherwise than this plan pins them, or any node naming a
    module the package does not have."""
    page = (ROOT / "site/reproduce/architecture.qmd").read_text(encoding="utf-8")
    block = _mermaid(page)
    assert block.count(ARCHITECTURE_VIS_SUBGRAPH) == 1
    edges = {
        (match.group(1), match.group(2))
        for match in re.finditer(r"^\s+(\w+) --> (\w+)$", block, re.MULTILINE)
    }
    assert {e for e in edges if _CHANGED_NODES & set(e)} == ARCHITECTURE_EDGES
    names = _diagram_modules(block)
    assert "_figures" in names
    assert [name for name in names if not _in_package(name)] == []
    flat = " ".join(page.split())
    for sentence in ARCHITECTURE_SENTENCES:
        assert sentence in flat


README_PACKAGE_ENTRIES = (
    "types.py", "config.py", "results.py", "api.py", "physics.py", "noise.py",
    "inference.py", "viz.py", "_style.py", "_support.py", "_replay.py",
    "_figures.py", "_clips.py", "_video.py", "_site_media.py", "_scenarios.py",
    "_echo_study.py", "_demos/",
)  # fmt: skip


def test_the_readme_structure_names_the_package_as_it_is() -> None:
    """Only this catches the README's module list naming a file the package does
    not have, or losing one it lists."""
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    listing = [line for line in _block(lines, "src/clocks/") if line.strip()]
    entries = tuple(line.split()[0] for line in listing)
    assert entries == README_PACKAGE_ENTRIES
    assert all((ROOT / "src" / "clocks" / entry).exists() for entry in entries)
    viz_line = (
        "    viz.py         Public facade over _figures, _clips, _echo_study, "
        "_replay and _style"
    )
    assert viz_line in listing
```

  (`_block` is the module's existing helper; `re` is already imported.)

- [ ] **Step 3: Run** `uv run pytest tests/test_api.py tests/test_generated_assets.py -q`. Expected: `test_viz_exports_exactly_the_documented_names` fails listing the nineteen extra names; `test_the_removed_names_are_gone` fails on `animate_echolocation`; the Architecture test fails on the subgraph (the page holds `_animate`, `_panels` and `_panels3d`); the README test fails on the entries (`_panels3d.py` listed, five missing).

- [ ] **Step 4: Implement.**
  - `git rm` the three modules and `tests/test_viz.py`. `src/clocks/viz.py` becomes:

```python
"""Figures, clips and the shared style (public facade).

Implementations live in ``clocks._figures`` (figures drawn from replays),
``clocks._echo_study`` (the range-study figure), ``clocks._clips`` (clips drawn
from replays), ``clocks._replay`` (the archive loader) and ``clocks._style``.
"""

from clocks._clips import render_clip
from clocks._echo_study import range_study_figure
from clocks._figures import density_figure
from clocks._replay import load_replay
from clocks._style import THEMES, Palette, palette, show, style

__all__ = [
    "THEMES",
    "Palette",
    "density_figure",
    "load_replay",
    "palette",
    "range_study_figure",
    "render_clip",
    "show",
    "style",
]
```

  - `README.md`, "Project structure": the `viz.py` line becomes `    viz.py         Public facade over _figures, _clips, _echo_study, _replay and _style`; the `_panels3d.py` line goes; after `_support.py` and before `_scenarios.py` come `    _replay.py     Replay archives: record, validate, save, load, compare`, `    _figures.py    Static figures drawn from a replay, and the trace helpers`, `    _clips.py      Clips drawn from replays (_clip_*.py hold the six builders)`, `    _video.py      Raw frames to H.264 through FFmpeg, and the decode check` and `    _site_media.py The site's clips and posters, drawn before Quarto renders`. The other lines stand.
  - `site/reproduce/architecture.qmd`: the Mermaid block becomes the following (the first 26 lines and the edges among `contracts`, `physnoise` and `core` are today's; every edge was read from the modules' imports at this plan's head, after Task 3; like today's, the diagram shows a selection of each module's imports, not all of them):

````text
```{mermaid}
flowchart TB
    subgraph contracts["Validation and data contracts"]
        validation["_validation"]
        types
        config
        results
    end
    subgraph physnoise["Physics and noise"]
        physics
        noise
        support["_support"]
    end
    subgraph core["Inference and public API"]
        inference
        api
    end
    subgraph vis["Visualization"]
        viz
        figures["_figures"]
        clips["_clips, _clip_*"]
        video["_video"]
        sitemedia["_site_media, _sources"]
        figstyle["_style, _fonts/"]
    end
    subgraph scenarios["Scenario tooling"]
        scenario["_scenarios"]
        echo["_echo_study"]
        replay["_replay"]
    end
    subgraph entry["Packaged entry points"]
        demos["_demos/*"]
    end
    console["uv run demo-*"]
    wrappers["scripts/demo_*.py"]

    types --> validation
    config --> validation
    config --> types
    results --> validation
    results --> config
    results --> types
    physics --> validation
    physics --> types
    support --> physics
    support --> types
    inference --> validation
    inference --> types
    api --> support
    api --> config
    api --> inference
    api --> noise
    api --> physics
    api --> results
    viz --> figures
    viz --> clips
    viz --> echo
    viz --> replay
    viz --> figstyle
    figures --> replay
    figures --> figstyle
    clips --> figures
    clips --> replay
    clips --> video
    clips --> figstyle
    sitemedia --> clips
    sitemedia --> replay
    sitemedia --> video
    sitemedia --> figstyle
    replay --> api
    replay --> config
    replay --> types
    scenario --> support
    scenario --> api
    scenario --> inference
    scenario --> physics
    echo --> scenario
    echo --> physics
    echo --> figstyle
    demos --> api
    demos --> scenario
    demos --> replay
    demos --> figures
    demos --> clips
    demos --> video
    demos --> figstyle
    console -. "pyproject entry points" .-> demos
    wrappers -. "thin imports" .-> demos
```
````

  and in "In words" the sentence "`viz` is the public facade over private animation and panel modules and the shared figure style (`_style`, with its bundled fonts)." becomes the first of `ARCHITECTURE_SENTENCES`. "The public/private boundary" says today that `clocks.__all__` holds "selected physics, noise, inference, and visualization names" (it holds no visualization name: read, `clocks.__all__` at `6e0527c`) and directs every library caller to `clocks`; its paragraph becomes: "The package's promised surface is the set of names curated in `clocks/__init__.py::__all__` (the API entry points, configuration and result dataclasses, core types, and selected physics, noise and inference names) together with `clocks.viz.__all__`: the figures, the clip renderer, the replay loader and the shared style. Entire modules are *not* the promised surface." followed by the second of `ARCHITECTURE_SENTENCES`. The page's other text stands.

- [ ] **Step 5: Tests pass; full gate** (the suite loses `tests/test_viz.py`'s 39 tests); **render the site and run both checks; the wheel check** on `dist/clocks-0.1.0-py3-none-any.whl`.

- [ ] **Step 6: Commit and push** as `Remove the GIF animators and panel functions; clocks.viz keeps nine names`.

---

### Task 6: Version 0.2.0 and the upgrade note

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `README.md`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests** in `tests/test_api.py` (beside Task 5's lists; `tomllib` imported, `ROOT` the repository root):

```python
UPGRADE_NOTE = (
    "## Upgrading from 0.1 "
    "Version 0.2.0 changes what the demos write and removes the old plotting "
    "names. This note covers everything since 0.1. "
    "- Removed from `clocks.viz`: the six `animate_*` functions, the eleven "
    "`plot_*` functions and the two `create_*_dashboard` functions. Added since "
    "0.1: `load_replay`, `render_clip`, `density_figure`, `range_study_figure`. "
    "- `--output PATH` becomes `--output-dir DIR`; the file names are the "
    "demo's, and options can no longer be abbreviated. "
    "- Each demo writes a replay archive and a PNG where six of them wrote a "
    "GIF; an MP4 is opt-in for those six, with `--video`. `demo-density` "
    "already wrote a PNG and now writes its archive beside it."
)


def test_the_version_is_0_2_0_in_the_project_and_the_lock() -> None:
    """Only this catches a version bump without `uv lock`, which fails CI's
    `uv sync --locked` (K2 of the commands plan)."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    [entry] = [package for package in lock["package"] if package["name"] == "clocks"]
    assert (project["version"], entry["version"]) == ("0.2.0", "0.2.0")


def test_the_readme_has_the_upgrade_note() -> None:
    """Only this catches the upgrade note changed or dropped; the counts it
    states are the lengths of the lists of removed names above."""
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    assert UPGRADE_NOTE in readme
    assert (len(ANIMATE), len(PLOT), len(DASHBOARD)) == (6, 11, 2)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_api.py -q`: both fail on their assertions (0.1.0; no note).

- [ ] **Step 3: Implement.** `pyproject.toml`: `version = "0.2.0"`. Then `uv lock`, and confirm `git diff uv.lock` changes exactly two lines: the `clocks` entry's `version` and the `[options]` `exclude-newer` timestamp (K2, decision C13); if any package entry changes, stop and report. `README.md` gains, after "## Run the demos" and before "## Run tests":

```markdown
## Upgrading from 0.1

Version 0.2.0 changes what the demos write and removes the old plotting names.
This note covers everything since 0.1.

- Removed from `clocks.viz`: the six `animate_*` functions, the eleven `plot_*`
  functions and the two `create_*_dashboard` functions. Added since 0.1:
  `load_replay`, `render_clip`, `density_figure`, `range_study_figure`.
- `--output PATH` becomes `--output-dir DIR`; the file names are the demo's,
  and options can no longer be abbreviated.
- Each demo writes a replay archive and a PNG where six of them wrote a GIF; an
  MP4 is opt-in for those six, with `--video`. `demo-density` already wrote a
  PNG and now writes its archive beside it.
```

- [ ] **Step 4: Tests pass; full gate; `uv lock --check`** exits 0; **the wheel check** on `dist/clocks-0.2.0-py3-none-any.whl`.

- [ ] **Step 5: Commit and push** as `Version 0.2.0, with an upgrade note`.

---

### Task 7: Mutation checks

The run changes no file; a fix it leads to is made in the task that owns the code (Step 3). **The coordinator runs it, with a mutation harness in worktrees that no reviewer or implementer is using**; mutants may run in several worktrees at once, one at a time in each, and no row is dropped or merged. In each worktree the environment is synced once on the clean tree (`uv sync --frozen`); for every mutant the harness sets `PYTHONDONTWRITEBYTECODE=1`, deletes every `__pycache__`, confirms `git status --porcelain` is empty, applies the mutant and restores it in a `try`/`finally`, and runs the whole suite, default and slow tests together, with `uv run --no-sync pytest -q -m ''`. Each run has a time limit of three times the clean baseline's duration, measured in Step 1; the harness starts each run in a new process group and kills the group when the limit passes (pull request 3's D47). No row is declared knowing that it hangs.

**The expected sets below are the declaration for this pull request's head.** Before the first mutant the coordinator records the baseline (every test passing, and the list of collected IDs) and checks each name in the table against it; a name that does not exist is corrected, with a note, before any mutant runs. Every mutant breaks library code, a script, a page or a configuration file, never a test or a test helper. "Each" means every parametrized case of that test: `[<demo>]` over the seven demos, the six clip demos or the five filter clip demos as the test is parametrized, and `test_demo_main_writes_its_archive_and_png[<demo>-<theme>]`. A cell does not claim that its tests are the only tests of that behaviour, only that these and no others fail for this one edit.

| Task | Mutation (the exact edit) | Tests that must fail, and no others |
|---|---|---|
| 1 | `_feed`: `complete = False` in the write's `except BrokenPipeError` deleted (the `break` stays) | `test_an_encoder_that_stops_reading_early_is_an_error[write_breaks]` |
| 1 | `_feed`: `complete = False` in the close's `except BrokenPipeError` becomes `pass` | `test_an_encoder_that_stops_reading_early_is_an_error[close_breaks]` |
| 1 | `_run_encoder`: the `if status == 0 and not complete:` check and its `raise` deleted | `test_an_encoder_that_stops_reading_early_is_an_error[write_breaks]`, `[close_breaks]` |
| 1 | `encode`: the cleanup is `temporary.unlink(missing_ok=True)` alone, with no `try` and no note | `test_a_failed_cleanup_keeps_the_encoders_error` |
| 2 | `write_outputs`: `os.replace(staged[1], finals[1])` added right after `_stage_png(...)`, and in the rename loop `os.replace(path, final)` made only `if final != finals[1]` | `test_a_failed_clip_leaves_every_existing_output_as_it_was`; `test_a_clip_that_does_not_decode_to_every_frame_is_not_published[short]`, `[undecodable]`; `test_a_failed_rename_removes_the_staged_files` (the PNG's rename now fails before the archive's) |
| 2 | `write_outputs`: `os.replace(staged[0], finals[0])` added right after `save_replay(...)`, and in the rename loop `os.replace(path, final)` made only `if final != finals[0]` | `test_a_failed_clip_leaves_every_existing_output_as_it_was`; `test_a_failed_png_leaves_every_existing_output_as_it_was`; `test_a_clip_that_does_not_decode_to_every_frame_is_not_published[short]`, `[undecodable]` |
| 2 | `write_outputs`: the loop that unlinks the staged names before staging deleted | `test_no_output_is_written_through_a_link_or_onto_a_leftover[staged_png_link]`, `[staged_png_file]` |
| 2 | `write_outputs`: each rename is `shutil.copyfile(path, final)` then `path.unlink()` in place of `os.replace(path, final)` (with `shutil` imported) | `test_no_output_is_written_through_a_link_or_onto_a_leftover[final_png_link]`, `[final_archive_link]`; `test_a_failed_rename_removes_the_staged_files` (its `os.replace` stand-in is never asked for the PNG) |
| 2 | `write_outputs`: `except BaseException as error:` becomes `except EncoderError as error:` | `test_a_failed_png_leaves_every_existing_output_as_it_was`; `test_a_failed_rename_removes_the_staged_files` |
| 2 | `write_outputs`: the preflight loop refusing a directory at a final name deleted | `test_a_directory_at_an_output_name_stops_the_run_before_anything_changes` |
| 2 | `write_outputs`: the rename loop moved out of the `try`, after the `except` clause | `test_a_failed_rename_removes_the_staged_files` |
| 2 | `write_outputs`: the `Saved:` loop deleted, and `print(f"Saved: {final}")` placed after each `os.replace` inside the rename loop | `test_a_closed_output_stream_does_not_stop_publication`; `test_a_failed_rename_removes_the_staged_files` (the archive's `Saved:` line is printed before the PNG's rename fails) |
| 2 | `write_outputs`: the `_check_clip(...)` call deleted | `test_a_clip_that_does_not_decode_to_every_frame_is_not_published[short]`, `[undecodable]`; `test_the_clip_is_drawn_from_the_replay_in_the_theme` (its `probed` list stays empty) |
| 2 | `_check_clip`: the `try`/`except ValueError` round `probe` removed | `test_a_clip_that_does_not_decode_to_every_frame_is_not_published[undecodable]` |
| 2 | `write_outputs`: when `encoder is None` and an entry is at the clip's name, `clip.unlink()` before the `Kept:` line | `test_a_run_without_a_clip_keeps_an_earlier_one_and_says_so[earlier_clip]`, `[dangling_clip_link]` |
| 2 | `write_outputs`: the `Kept:` `print(...)` call replaced by `pass` | `test_a_run_without_a_clip_keeps_an_earlier_one_and_says_so[earlier_clip]`, `[dangling_clip_link]` |
| 2 | `write_outputs`: the `Kept:` condition reduced to `if encoder is None:` | `test_a_run_without_a_clip_keeps_an_earlier_one_and_says_so[no_clip]`; `test_write_outputs_writes_the_archive_and_png_into_a_new_directory[demo_1d]`, `[demo_density]` |
| 2 | `write_outputs`: `has_clip and ` deleted from the `Kept:` condition | `test_the_density_demo_says_nothing_of_a_clip` |
| 2 | `write_outputs`: `not final.is_symlink() and ` deleted from the directory preflight | `test_a_link_to_a_directory_at_an_output_name_is_replaced` |
| 2 | `_remove`: the `try`/`except OSError` round `path.unlink(missing_ok=True)` removed | `test_a_failed_cleanup_keeps_the_error_and_removes_the_rest` |
| 2 | `write_outputs`: `clip.is_symlink() or ` deleted from the `Kept:` condition | `test_a_run_without_a_clip_keeps_an_earlier_one_and_says_so[dangling_clip_link]` |
| 2 | `_stage_png`: `style(theme)` becomes `style("light")` | `test_the_png_is_the_figure_drawn_for_the_theme[dark]`; `test_demo_main_writes_its_archive_and_png` (each `-dark` case, seven) |
| 2 | `_stage_png`: `figure(replay, colors)` becomes `figure(replay, palette("light"))` (with `palette` imported) | `test_the_png_is_the_figure_drawn_for_the_theme[dark]` |
| 2 | `_stage_png`: `save_png(fig, file)` becomes `blank = Figure(figsize=fig.get_size_inches()); FigureCanvasAgg(blank); save_png(blank, file)` (with `from matplotlib.backends.backend_agg import FigureCanvasAgg` added; `Figure` is already imported) | `test_the_png_is_the_figure_drawn_for_the_theme[light]`, `[dark]`; `test_png_is_the_final_frame` (each, six); `test_density_command_saves_the_figure_it_drew` |
| 2 | `write_outputs`: `render_clip(replay, staged[2], encoder=encoder)`, the `theme` argument dropped | `test_the_clip_is_drawn_from_the_replay_in_the_theme` |
| 2 | `write_outputs`: the `directory.mkdir(...)` line deleted | `test_write_outputs_writes_the_archive_and_png_into_a_new_directory[demo_1d]`, `[demo_density]`; `test_the_default_run_writes_into_output` |
| 2 | `_stage_png`: the `try`/`finally` removed (the `with path.open(...)` block dedented into the `with style(...)` block) and `plt.close(fig)` placed after the `with path.open(...)` block | `test_a_failed_save_still_closes_the_figure` |
| 3 | `make_parser`: `allow_abbrev=False` removed | `test_output_is_no_longer_an_option` (each, seven); `test_options_cannot_be_abbreviated` |
| 3 | `add_common_arguments`: `parser.add_argument("--output-dir", "--output", ...)` | `test_output_is_no_longer_an_option` (each, seven) |
| 3 | `make_parser`: `allow_abbrev=False` removed, and `add_common_arguments` adds `--output` with an `action` whose `__call__` is `parser.error("--output is now --output-dir")` | `test_options_cannot_be_abbreviated` |
| 3 | `add_common_arguments`: the default of `--observations` is `10` | `test_demo_parser_defaults` (each, seven) |
| 3 | `add_common_arguments`: the default of `--output-dir` is `Path(".")` | `test_demo_parser_defaults` (each, seven); `test_the_default_run_writes_into_output` |
| 3 | `demo_2d._run_demo`: `Path(".")` passed to `write_outputs` in place of `args.output_dir` | `test_demo_main_writes_its_archive_and_png[demo_2d-light]`, `[demo_2d-dark]` |
| 3 | `run_command`: the `find_encoder()` lookup and its `try` moved after `run(args)`, so `args.encoder` is `None` during the run | `test_video_writes_the_clip` (each, six); `test_video_without_an_encoder_stops_before_the_run` (each, six); `test_a_failed_encode_ends_with_a_message_not_a_traceback` |
| 3 | `run_command`: the `try`/`except EncoderNotFound` round `find_encoder()` removed | `test_video_without_an_encoder_stops_before_the_run` (each, six) |
| 3 | `run_command`: the `try`/`except EncoderError` round `run(args)` removed | `test_a_failed_encode_ends_with_a_message_not_a_traceback` |
| 3 | `run_command`: `notes = ()` in place of `getattr(error, "__notes__", ())` | `test_a_failed_encode_ends_with_a_message_not_a_traceback` |
| 3 | `_replay_embed_violations` (`tests/test_site_figure_style.py`): `_COMMAND_CLIP.sub("", line).lower()` back to `line.lower()` | `test_replay_embed_violations[a_command_output_clip]`; `test_no_page_has_unpinned_video_markup[site/reproduce/getting-started.qmd]` |
| 3 | `_COMMAND_CLIP`: the lookbehind `(?<![\w./-])` removed | `test_replay_embed_violations[an_output_clip_under_another_directory]` |
| 3 | `demo_model_comparison._run_demo`: `print_truth(replay)` replaced by `print(f"Truth: x1={TRUE_X1}, x2={TRUE_X2}, M1={TRUE_M1}, M2={TRUE_M2}")` and `print(f"True rates: {scenario().true_rates}")` | `test_model_comparison_command_prints_from_the_replay_and_records_once` |
| 3 | `demo_1d._run_demo`: `record(n_observations=args.observations)`, `n_particles` not passed | `test_demo_main_writes_its_archive_and_png[demo_1d-light]`, `[demo_1d-dark]`; `test_command_prints_the_recorded_run[demo_1d]` |
| 3 | `demo_2d._run_demo`: `theme="light"` passed to `write_outputs` | `test_demo_main_writes_its_archive_and_png[demo_2d-dark]` |
| 3 | `demo_1d._run_demo`: the figure passed is `lambda replay, colors: build_clip(replay, colors).fig` (`build_clip` imported; for `demo_1d` that figure already shows the last row, `_clip_tracks.update` being called before it returns, so the mutant is a different call that bypasses `final_frame`, caught by its spy, not a different picture) | `test_png_is_the_final_frame[demo_1d]` |
| 3 | `demo_multi_mass._run_demo`: `encoder=None` passed to `write_outputs` | `test_video_writes_the_clip[demo_multi_mass]` |
| 3 | `print_estimate`: `meta['n_particles']` in the ESS line replaced by `1000` | `test_command_prints_the_recorded_run` (each, five) |
| 3 | `print_estimate`: `d = trace_decimals(prior_range)` | `test_command_prints_the_recorded_run` (each, five) |
| 3 | `demo_model_comparison._run_demo`: `record(...)` called a second time, its result unused | `test_model_comparison_command_prints_from_the_replay_and_records_once` |
| 3 | `demo_density.build_parser`: `add_video_argument(parser)` added | `test_density_has_no_video_option` |
| 3 | `demo_2d.py`: `import clocks.viz` added after the other imports | `test_no_demo_module_imports_the_facade_or_the_old_drawing_code` |
| 3 | `README.md`: the `demo-2d` line's comment is `# → output/demo_2d.gif` | `test_the_command_block_is_the_commands[README.md]` |
| 3 | Getting Started: a line `uv run demo-density --video` added inside the block | `test_the_command_block_is_the_commands[site/reproduce/getting-started.qmd]` |
| 3 | the Reproducibility page: "With `--video`, the six also write the clip, drawn as the site's are." becomes "`--video` also writes the clip." | `test_the_commands_are_described_as_they_are` |
| 3 | `README.md`: "All seven, with commentary: [jbwhit.github.io/clocks](https://jbwhit.github.io/clocks/)." added at the end of the second paragraph of "Run the demos" | `test_readme_describes_the_site_and_the_commands_as_they_are` |
| 4 | `bare_environment`: `"PYTHONPATH"` removed from `_INHERITED_WAYS_IN` | `test_the_bare_environment_reaches_no_checkout_and_no_encoder` |
| 4 | `bare_environment`: `"IMAGEIO_FFMPEG_EXE"` removed from `_INHERITED_WAYS_IN` | the same |
| 4 | `bare_environment`: `PATH` becomes `os.pathsep.join([str(scripts_directory), inherited.get("PATH", "")])` | the same |
| 5 | `clocks.viz`: `from clocks._clips import final_frame`, and `"final_frame"` added to `__all__` | `test_viz_exports_exactly_the_documented_names` |
| 5 | `clocks.viz`: `from clocks._clips import build_clip as animate_inference` added, not in `__all__` | `test_the_removed_names_are_gone` |
| 5 | the Architecture page: a line `        animate["_animate"]` added to the `vis` subgraph | `test_the_architecture_page_maps_the_drawing_modules_as_they_are` |
| 5 | the Architecture page: an edge `    viz --> video` added | `test_the_architecture_page_maps_the_drawing_modules_as_they_are` |
| 5 | the Architecture page: the boundary sentence's "and the figure, clip and style names from `clocks.viz`" deleted | `test_the_architecture_page_maps_the_drawing_modules_as_they_are` |
| 5 | `README.md`: `    _panels3d.py   3D plotting primitives for the echolocation dashboard` put back after `_support.py` | `test_the_readme_structure_names_the_package_as_it_is` |
| 6 | `pyproject.toml`: `version = "0.1.0"`, the lock left | `test_the_version_is_0_2_0_in_the_project_and_the_lock` |
| 6 | `uv.lock`: the `clocks` entry's `version = "0.1.0"`, the project left | the same |
| 6 | `README.md`: the upgrade note's `--output` bullet removed | `test_the_readme_has_the_upgrade_note` |

What the table rests on. Every expected set is derived from the tests and code as this plan specifies them; none was run (Task 1's two tests were run red and green while planning, not under mutation). The least certain rows, read first as findings about the plan if they differ: the `style("light")` row, which assumes a final frame's corner pixel comes from the face colour that `style` sets and not from the palette (K8 measured the light and dark corners, not this mutant), and the `palette("light")` row, which assumes the reverse (no clip builder sets the figure's face colour from `colors`; read in `_clip_*.py`, where only an inset and the 3D panes are filled from `colors.background`); and the `run_command` rows, which depend on `pytest.fail` raising through `main` (it raises `Failed`, a `BaseException`, which `pytest.raises(SystemExit)` does not catch). Behaviours with no mutant, and why: the exclusive `open("xb")` of the staged PNG (it guards a race between the leftover's removal and the open, which no test reproduces); the wheel check's two runs (decision C12: CI runs them); `ascii_name` (its effect is pinned through `test_command_prints_the_recorded_run`, which spells the names independently).

- [ ] **Step 1: Complete and check the declaration** against the baseline's collected IDs, and measure the baseline's duration for the time limit.
- [ ] **Step 2: Run each mutation** and record the exact set of failing test IDs and any change in what is collected.
- [ ] **Step 3: Class every discrepancy, and resolve it**, exactly as pull request 3's Task 14, Step 3 says: an extra failure, a missing predicted failure, a surviving mutant, a collection failure, a timeout or an unrelated crash; the declared set stays beside the observed one, and no test is changed merely so that the two agree. A correction of an expected set stays in this task; a fix to code or tests reopens the task that owns it, reruns that task's gates (and the site build and checks where it changes a page), and restarts the mutation run from a clean baseline pinned to the new head (pull request 3's D48).
- [ ] **Step 4: Report** the table of results with each row's class and resolution. The run counts as evidence only when every row is exact or resolved; until then the hand-over's mutation line says "not complete" and lists what is open.

---

### Hand-over

The coordinator opens the pull request as a draft and gives in its description: what each command writes now and what happens to an older clip; that `--output` is gone and options cannot be abbreviated; the `encode` fix, the decode check before a clip is published, and K6; the upgrade note; the deferred items taken and left (decision C16); the mutation table with its declared and observed sets and, for each discrepancy, its class and resolution (Task 7, Steps 3 and 4). No embed, clip or poster changes, so no browser check is owed. **Before the merge**, as in pull request 3's hand-over: the merge gate is Codex's verdict on the reviewed head with green checks on that same head. After review converges and the pull request is marked ready, the coordinator finds, by the head's full SHA, the run each merge-gating workflow started for it and confirms that each concluded `success`: `CI` (its default and slow steps, and the wheel check with its two new `ok:` lines); and `Site check` on Quarto 1.11.9, with `Figure contract OK: 10 pages, 18 figures checked.` `Replay report` is a diagnostic: its steps continue on error, so its conclusion says only that it ran; the coordinator reads its three jobs' reports and states in the pull request whether each demo reproduced, without treating a `success` conclusion as that evidence. **After the merge**, the `CI` and `Deploy site` runs for the merge commit on `main` are confirmed the same way.

## Self-review

**The design's section 7, line by line.** Commands record and print from the replay, and write an archive and a PNG (Task 3, through Task 2's writer); `--video` (Tasks 2 and 3); `--output-dir` replacing `--output` (Task 3, decision C4); `--theme` (Task 3); no `--video` for density (Task 3); the PNG needs only Matplotlib and video is opt-in, so the default works from a bare wheel (Task 4). The break documented: version 0.2.0 and the upgrade note with the three removals, the option change and what the demos write (Tasks 5 and 6). The wheel check's added run (Task 4). README and Getting Started show the new commands (Task 3). The Architecture page's module list (Task 5). Already done by pull request 3 and not repeated: the README's GIF and its paragraph about committed media, and `docs/someday-maybe.md`'s remark about static GIFs.

**Where the design and the earlier draft disagreed with the code.** The design's section 7 puts the version and the note in pull request 3 (C1). The earlier draft: closed every PNG's figure as if all were pyplot's (only `density_figure`'s is; `final_frame`'s has its own canvas, and `plt.close` on it is harmless); saved the PNG to a staged path rather than into an exclusively created file, as `_site_media` stages its posters; mapped `μ` and `σ` in `ascii_name`, which no printer needs; said a failing encoder shows as `BrokenPipeError` turned into `EncoderError` (the broken pipe is swallowed and the status decides, so status 0 was trusted: K6, Task 1); expected `uv lock` to change only the version line (it moves the `exclude-newer` timestamp too: K2); and cited README lines 108 and 111 (now 106 and 109).

**What this plan removes of pull request 3's.** The README's and the Reproducibility page's sentences that the commands still write GIFs, with the assertions that pinned them (Task 3). Nothing else of pull request 3 is changed apart from `encode` (Task 1): no archive, no clip, no embed, no site-media code.

**Each task's tests at its own commit.** Task 1's tests use the module's own fake process and stand-ins. Task 2's call `write_outputs` directly, with the commands still on the GIF path. Task 3 changes the commands, their tests and every sentence about them in one commit. Task 4's test loads the script by path. Task 5's run after Task 3 has taken the demos off the old modules. Task 6's note is true once Task 5 has removed the names.

**Size.** Seven tasks: one deferred fix, the writer, the commands, the wheel check, the removals, the release and the mutation run. The writer is its own task so that its failure modes are tested directly, not through seven commands.

**Names.** `STAGED`, `write_outputs(replay, directory, figure, *, theme="light", encoder=None) -> list[Path]`, `make_parser(description)`, `add_common_arguments(parser, *, observations, particles)`, `add_video_argument(parser)`, `run_command(parser, run, argv) -> int`, `ascii_name`, `print_truth`, `print_estimate` and `bare_environment(scripts_directory, inherited)` are used with these signatures in every task and test.
