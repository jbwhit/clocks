# Replay Commands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the seven `demo-*` commands record a run, print it from the replay and write an archive and a PNG, with the clip on request; then remove the GIF animators, the panel functions and their public names, and release the result as version 0.2.0 with an upgrade note.

**Architecture:** `clocks._demos._common` gains one parser factory, one runner and one writer shared by the seven commands. The writer stages every output of a run beside its destination and moves nothing into place until all are staged. The six clip commands draw their PNG with `clocks._clips.final_frame` and their MP4 with `render_clip`, both from pull request 3; `demo-density` keeps `density_figure`. With no caller left, `_animate.py`, `_panels.py`, `_panels3d.py` and `tests/test_viz.py` are deleted and `clocks.viz` exports nine names.

**Tech Stack:** Python 3.12+, NumPy, Matplotlib 3.10, FFmpeg through `imageio-ffmpeg` (a development dependency) or `ffmpeg` on `PATH`, pytest, uv, ruff, Quarto 1.11.9 on the runner, GitHub Actions.

**Spec:** docs/plans/2026-11-19-replay-and-animation-design.md

**Status:** Draft. To be revised against the merged pull request 3 and reviewed then.

This is pull request 4. The spec's section 11 has three pull requests; its third was split after the site work, by the coordinator's ruling (decision C1). `docs/plans/2026-11-24-replay-clips.md` is pull request 3 and must be merged first: this plan consumes `clocks._video`, `clocks._clips` and `frame_rgba` as that plan defines them, and every statement here about "the code as pull request 3 leaves it" is read from that plan, not from a tree, and is to be checked against the merged tree before review. The spec's section 7 binds this plan, with the "Site" part of "Testing" that names `tests/test_cli.py`. Round 1 of the review of the single plan (Codex xhigh at `1c3a6b5`) found two Blocking faults in what is now this plan; both are fixed below (decisions C3 and C5). Round 2 (at `906eb19`) found that the publication spy of Task 1 would also have counted the renames that `save_replay` and the encoder make while staging; it now looks at destinations. Round 3 (at `c292181`) confirmed that fix and raised nothing else here; its rule for classing mutation discrepancies (pull request 3's Task 14, Step 3) is adopted by Task 3 below.

## Global Constraints

- Python only through uv: `uv run pytest`, `uv run ruff format .`, `uv run ruff check .`, `uv run python ...`. Never bare `python` or `pip`.
- **Every task runs the same cycle**: write the tests of Step 1; run them and see them fail for the stated reason; implement; run them and see them pass; run the full local gate (`uv run ruff format --check .`, `uv run ruff check .`, `uv run pytest -q`, `uv run pytest -q -m slow`) and read its output; then, in a separate command, commit with the task's message and push. Do not open a pull request (the coordinator does). Begin every git step with an explicit `cd` to the worktree.
- **Every task that changes a page also renders the site and runs both checks**, after the gate and before the commit: `cd site && uv run --frozen quarto render && cd ..`, then `uv run --frozen python scripts/check_site_links.py` and `uv run --frozen python scripts/check_site_figures.py`, which must print `Figure contract OK: 10 pages, 18 figures checked.`
- Commit messages are passed with a quoted heredoc (`<<'MSG'`) and end with a `Co-Authored-By:` trailer naming the model that wrote the commit.
- Implementers never use `git checkout`, `git restore` or `git stash` to undo anything. Task 3's mutants are run by the coordinator's harness in separate worktrees.
- **A red step fails on behaviour, not on a missing name**: where a task's tests import names that do not exist yet, its Step 2 first adds them with bodies that do nothing useful.
- **The repository is public.** No committed file may contain an absolute local path, a person's name or an email address; the one exception is the `Co-Authored-By:` trailer, which names a model.
- **A long job runs in the foreground** with `timeout: 600000`; if it needs longer, wait in a bounded foreground loop, `until ! pgrep -f '[p]ytest -q' >/dev/null; do sleep 10; done; tail -20 <log>`, and call again; never end a turn on a background job. The bracket in the pattern is load-bearing.
- **Commands** (design, section 7): `uv run demo-2d` writes `output/demo_2d.npz` and `output/demo_2d.png`; `--video` adds `output/demo_2d.mp4`; `--output-dir` replaces `--output`; `--theme` is new; `demo-density` has no `--video`; the PNG is the final frame (for `demo-density`, the density figure) and needs only Matplotlib; `--video` needs an encoder, which `uv sync` provides in a checkout, and without one the command stops and says how to get one. The version becomes 0.2.0.
- **Removed from `clocks.viz`** (section 7): the six `animate_*` functions, the eleven `plot_*` functions and the two `create_*_dashboard` functions.
- **No golden encoded bytes, and nothing computed compared exactly across machines**, as in pull request 3's plan.
- **No archive is re-recorded and no page's clip changes.** `git diff --stat main -- assets/replays site/story site/index.qmd` is empty at the end.

## Review Focus

1. **`--output` still parses.** `--output` is an unambiguous prefix of `--output-dir`, and argparse accepts prefixes by default: a parser that merely drops the option still takes `--output x` and writes into a directory called `x` (measured, K1). Task 1.
2. **A run that fails part-way over an existing set of outputs.** An encoder that dies after the archive and the PNG are written must not leave a new archive beside an old clip under the final names, a failed PNG save must not truncate the PNG that was there, and a destination that is a symbolic link must be replaced, not written through. Task 1.
3. **An older clip beside a run made without `--video`.** The clip then belongs to an earlier run than the archive and PNG beside it. It is kept, and the command says so in one line. Task 1.
4. **A wheel installed with no encoder anywhere.** The default command must work, and `--video` must fail with instructions, with `PATH`, `PYTHONPATH` and `IMAGEIO_FFMPEG_EXE` unable to supply an encoder by accident. Task 1.
5. **`import clocks.viz` after the deletions.** Nothing may still import `_animate`, `_panels` or `_panels3d`: the six demo modules stop in Task 1, and Task 2 deletes the modules only after a search built from the whole list of removed names. Task 2.

## Decisions this plan makes

- **C1. A fourth pull request.** Section 11 says three. Its third is split after the site work: pull request 3 is the encoder, the clips and the site; this one is the commands, the public names, the version and the removals. Between the two merges the commands write GIFs in the old layout while the site shows clips, and pull request 3's README and Reproducibility page say so in two pinned sentences; Task 1 here deletes those sentences and their assertions in the commit that makes them false. Cost: two animation code paths in the repository for that time.
- **C2. The commands change before the old modules go**, in two tasks: Task 1 leaves `_animate.py` with no caller, Task 2 deletes it. Each commit imports cleanly.
- **C3. Long options cannot be abbreviated** (Blocking 2 of round 1). Every command's parser is made by one factory with `allow_abbrev=False`, so `--output` is an error as the design's "replaces" requires, and so is `--obs`. Cost: a user who abbreviated `--observations` must spell it.
- **C4. `--video` with no encoder stops before inference**, through `parser.error`: exit status 2, a message naming `imageio-ffmpeg` and `ffmpeg`, nothing written. The design says the command "stops and says how to get one" without saying when.
- **C5. Outputs are staged, then published together** (the coordinator's ruling on Blocking 3). Each output of a run is first written beside its destination under a staged name, the final name with `.staged-` in front: the archive with `save_replay`, the PNG with `save_png`, the clip with `render_clip`. Nothing is moved into place until every output of that run is staged; a failure before then removes the staged files and leaves every existing file as it was. Publication is one `os.replace` per file, archive, PNG, clip; a destination that is a symbolic link is replaced, never written through. Stated limit: those two or three renames are not one atomic step, so a failure between them, which needs the file system itself to fail, can leave files of two runs side by side; the command then exits with the error.
- **C6. A run without `--video` keeps an existing `<demo>.mp4` and says so** (the coordinator's ruling): one line, `Kept: <path> is from an earlier run and was not updated (--video writes a new one)`. Cost recorded: a stale clip can sit beside a newer archive, disclosed only at run time.
- **C7. `demo-density` writes its archive too**, with `--output-dir` and `--theme` and no `--video`. Section 7 says each command "records its run" and writes an archive and a PNG. It already wrote a PNG; what is new for it is the archive and the fixed file name. Pull request 2's tests that it "writes only its PNG" become tests that it writes exactly those two files.
- **C8. The old names go at once**, with no deprecation period: section 7 calls it a documented break. `clocks.viz` loses the nineteen names and keeps nine.
- **C9. The upgrade note describes the whole release, 0.1 to 0.2.0**, not this pull request alone: of the four names it lists as added, `load_replay`, `density_figure` and `range_study_figure` arrived in pull request 2 and `render_clip` in pull request 3.
- **C10. The commands print one more decimal than the trace headers** (`trace_decimals(prior range) + 1`: three for positions, as today, and four for masses), because review of a re-recording rests on the printed estimates (design, "Risks"). `demo-density`'s printing is left as pull request 2 pinned it.
- **C11. The wheel check stays a script without a unit test.** It builds an environment and installs a wheel, which a unit test should not; CI's "Smoke-test installed entry points" step runs it on every run, and Tasks 1 and 2 run it by hand. Its own logic therefore has no mutant in Task 3; the behaviour it checks (the commands with and without an encoder) has unit tests that do.
- **C12. Output is ASCII apart from `±`**, as today: parameter names are printed through `ascii_name` (`x₁` as `x1`, `μ` as `mu`).

## Measurements

Run on 2026-11-24 in the pull-request-3 worktree at `177a425` (that is, on `main` before pull request 3), macOS arm64, Python 3.14.0. Each must be run again on the merged pull request 3.

- **K1. Abbreviation.** `argparse.ArgumentParser()` with only `--output-dir`: `parse_args(["--output", "x"])` gives `Namespace(output_dir='x')`. With `allow_abbrev=False` the same call exits with status 2, "unrecognized arguments: --output x". The seven parsers are built today with `argparse.ArgumentParser(description=__doc__)`, the default.
- **K2. The project's version is in the lock.** `uv.lock` holds `name = "clocks"`, `version = "0.1.0"`, `source = { editable = "." }`; CI installs with `uv sync --locked`.
- **K3. Who uses the old names.** The nineteen names are imported only by the six clip demo modules, `tests/test_viz.py` and `clocks.viz` itself. `_animate.py` imports `_panels` and `_panels3d`; nothing else imports any of the three. They are named in prose in `README.md` (lines 108 and 111) and `site/reproduce/architecture.qmd` (lines 29 to 31, 62 to 69 and 89). `tests/test_viz.py` has eighteen test classes, each on removed names only.
- **K4. Today's parser test asserts lower bounds.** `tests/test_cli.py`, `test_demo_parser_preserves_defaults_and_accepts_fast_overrides`: `assert defaults.observations > 1` and `assert defaults.particles > 16`. (Round 1 asked to "keep the existing assertions of the actual defaults"; there are none to keep. Task 1 adds them.)
- **K5. Small recordings.** `record(n_observations=1, n_particles=16)`, `(2, 16)` and `(9, 40)` succeed for the six clip demos in 0.02 to 0.18 s each; `(2, 1)` succeeds for all six.
- **K6. A failing encoder** shows up as `BrokenPipeError` on a frame write, which pull request 3's `encode` turns into `EncoderError`.

Not measured: the wheel run with no encoder (Task 1, Step 4, and CI); every new test.

---

### Task 1: The demo commands: `--output-dir`, `--theme`, `--video`

**Files:**
- Modify: `src/clocks/_demos/_common.py` and the seven modules beside it; `scripts/check_wheel_entrypoints.py`; `README.md`; `site/reproduce/getting-started.qmd`; `site/reproduce/reproducibility.qmd`
- Test: `tests/test_cli.py`, `tests/test_density_demo.py`, `tests/test_generated_assets.py`

**Interfaces:**
- Consumes, as pull request 3 leaves them: `final_frame`, `render_clip`, `CLIP_HEIGHTS` from `clocks._clips`; `find_encoder`, `EncoderNotFound`, `EncoderError` from `clocks._video`; `save_replay`, `Replay` from `clocks._replay`; `style`, `save_png`, `THEMES`, `Palette` from `clocks._style`; `trace_decimals`, `density_figure` from `clocks._figures`.
- Produces, in `src/clocks/_demos/_common.py`:

```python
STAGED = ".staged-"

def make_parser(description: str | None) -> argparse.ArgumentParser: ...
def add_common_arguments(parser: argparse.ArgumentParser, *, observations: int,
                         particles: int) -> None: ...
def add_video_argument(parser: argparse.ArgumentParser) -> None: ...
def run_command(parser: argparse.ArgumentParser,
                run: Callable[[argparse.Namespace], None],
                argv: Sequence[str] | None) -> int: ...
def ascii_name(name: str) -> str: ...
def print_truth(replay: Replay) -> None: ...
def print_estimate(replay: Replay) -> None: ...
def write_outputs(replay: Replay, directory: Path, theme: str,
                  figure: Callable[[Replay, Palette], Figure],
                  encoder: str | None = None) -> list[Path]: ...
```

  - `make_parser(description)` returns `argparse.ArgumentParser(description=description, allow_abbrev=False)`. Every `build_parser()` uses it.
  - `add_common_arguments` adds `--output-dir DIR` (a `Path`, default `Path("output")`), `--observations`, `--particles` (as today, with the defaults passed in) and `--theme` (choices `light` and `dark`, default `light`). There is no `--output`. `add_video_argument` adds `--video` (`store_true`), whose help says it also writes `<demo>.mp4` and needs an FFmpeg encoder.
  - `run_command` parses `argv`; sets `args.encoder` to `None`, or, when the parser has `--video` and it was given, to `find_encoder()`, calling `parser.error(str(error))` on `EncoderNotFound` (exit status 2, before `run` is called); calls `run(args)`; turns an `EncoderError` from it into `parser.exit(1, f"{parser.prog}: error: {error}\n")`; returns 0.
  - `ascii_name` maps `₁` and `₂` to `1` and `2`, `μ` to `mu` and `σ` to `sigma`.
  - `print_truth(replay)` prints `Truth: ` followed by `f"{ascii_name(name)}={value:g}"` for each parameter, joined by `, `, then `f"True rates: {replay.arrays['true_rates']}"`.
  - `print_estimate(replay)`, for a filter replay, prints `f"Final estimate after {T} observations:"`, then per parameter `f"  {ascii_name(name)} = {mean:.{d}f} ± {sd:.{d}f}  (true: {truth:g})"` with the last row's values and `d = trace_decimals(prior range) + 1`, then `f"  ESS = {ess:.0f} / {n_particles}"`, all from the replay.
  - **`write_outputs`, exactly.** With `<demo>` from `replay.meta["demo"]`, it creates `directory` if missing, then:
    1. **stages**, in this order, each under the final name with `STAGED` in front: `<demo>.npz` with `save_replay`; `<demo>.png`, the figure `figure(replay, colors)` returns inside `with style(theme) as colors:`, with `save_png`, the figure closed in a `finally`; and, when `encoder` is given, `<demo>.mp4` with `render_clip(replay, staged, theme=theme, encoder=encoder)`. Until all of these exist nothing under a final name is touched. Any failure removes the staged files of this run and propagates.
    2. **publishes**: `os.replace(staged, final)` for the archive, then the PNG, then the clip, printing `f"Saved: {final}"` after each. `os.replace` replaces a symbolic link at the destination; it does not write through it.
    3. when `encoder` is `None` and `<directory>/<demo>.mp4` exists, prints `f"Kept: {path} is from an earlier run and was not updated (--video writes a new one)"`. The file is not removed.
    It returns the paths it published.
- Each of the six clip demos: `build_parser()` is `make_parser(__doc__)` with `add_common_arguments` and `add_video_argument`; `_run_demo(args)` is `replay = record(n_observations=args.observations, n_particles=args.particles)`, `print_truth(replay)`, `print_estimate(replay)`, `write_outputs(replay, args.output_dir, args.theme, final_frame, args.encoder)`; `main(argv)` is `return run_command(build_parser(), _run_demo, argv)`. `OUTPUT_PATH` and every import only the old path used go, the import from `clocks.viz` among them. The model-comparison command prints, in place of `print_estimate`: `f"True model: K={true_k}"` first; for k = 20, 40, … up to T, `f"After {k} observations:"` and one `f"  K={K}: posterior={p:.4f}"` per candidate; then `f"MAP model: K={best}"`, `f"Estimate: {final_mean}"`, `f"Std:      {final_std}"` and `f"ESS:      {ess:.0f} / {n_particles}"` for the candidate with the largest final probability, all from the replay. It records once.
- `demo_density`: the same, without `add_video_argument`, with `density_figure` as the figure; its printing is as pull request 2 left it.
- `scripts/check_wheel_entrypoints.py` gains, after the `--help` loop, a check that the commands work from the wheel alone. It runs in a new empty directory, with an environment built for the purpose: `PATH` is the temporary environment's scripts directory and nothing else, and `PYTHONPATH` and `IMAGEIO_FFMPEG_EXE` are removed, so neither a checkout nor an encoder can be reached by inheritance. First, **the default works with no encoder**: `demo-1d --output-dir out --observations 2 --particles 16` exits 0 and leaves `out/demo_1d.npz` and `out/demo_1d.png` and nothing else. Then, **the opt-in fails with instructions**: the same command with `--video` exits with status 2, its standard error holds `imageio-ffmpeg`, and `out/` is as the first run left it (no `demo_1d.mp4`, no staged name, the two files unchanged byte for byte). It prints one `ok:` line for each.

- [ ] **Step 1: Write the failing tests.** In `tests/test_cli.py`, `DEMO_CASES` holds module and executable names only; `CLIP_DEMOS` is the six with a clip. Every test that calls `main` runs with the working directory changed to `tmp_path` (`monkeypatch.chdir`) and, unless it tests the defaults themselves, passes `--observations` and `--particles` explicitly, so a regression can neither start a default run nor write into the checkout. "An existing set" is `<demo>.npz` holding `b"old archive"`, `<demo>.png` holding `b"old png"` and `<demo>.mp4` holding `b"old clip"` in the output directory before the run. `small(name)` is pull request 3's cached nine-observation recording.

  - `test_demo_parser_defaults` (each demo): `output_dir == Path("output")`, `theme == "light"`, `observations == module.scenario().n_observations`, `particles == module.scenario().n_particles`, and `video is False` for a clip demo. Today's test asserts only that the two counts exceed 1 and 16 (K4); this one ties each to the scenario the archive was recorded from. Only this catches a command whose default run is not the packaged one.
  - `test_demo_parser_overrides` (each demo): `--output-dir x --observations 1 --particles 16 --theme dark` parses to those values.
  - `test_output_is_no_longer_an_option` (each demo): `main(["--output", "x.png", "--observations", "1", "--particles", "16"])` raises `SystemExit` with code 2, the module's `record` (replaced by `pytest.fail`) is not called, and `tmp_path` is empty afterwards. Only this catches `--output` accepted as a prefix of `--output-dir` (Review Focus 1).
  - `test_options_cannot_be_abbreviated`: for `demo_1d`, `main(["--obs", "1", "--particles", "16"])` raises `SystemExit` with code 2. With the test above it shows the cause, not one spelling.
  - `test_demo_main_writes_its_archive_and_png` (each demo): `main(["--output-dir", "new/out", "--observations", "1", "--particles", "16"]) == 0`; `tmp_path / "new/out"` holds exactly `<demo>.npz` and `<demo>.png`; the archive loads with `meta["demo"] == <demo>`, `n_observations == 1` and `n_particles == 16`; the PNG's size is 1500 by twice the demo's declared height (1080 for `demo_density`). Only this catches a command that writes under another name, draws another run than it archived, does not create the directory, or leaves a staged name behind.
  - `test_theme_dark_draws_the_dark_png` (each demo): with `--theme dark`, the PNG's first pixel is `palette("dark").background`; without it, the light one. Only this catches `--theme` parsed and ignored.
  - `test_png_is_the_final_frame` (each clip demo): with `clocks._demos._common.save_png` wrapped by a spy and the module's `final_frame` wrapped by a spy, `save_png` is called once, with the figure `final_frame` returned (`is`) and a path named `.staged-<demo>.png`, and `final_frame` once, with the replay that was archived. Only this catches a PNG of a different figure.
  - `test_outputs_are_staged_before_any_is_published`: for `demo_1d` with `--video`, an existing set, and spies that append to one list: on `save_replay`, `save_png` and `render_clip` in `clocks._demos._common` (each noting `"stage"` and, at that moment, the bytes of the three final files), and a stand-in for `os.replace` that calls through every time and notes `"publish"` and the destination's name only when the destination is one of the three final paths (`save_replay` and the encoder each move a temporary file of their own into a staged name with the same function, and those moves are staging, not publication). The list is three stages, during each of which all three final files still hold their old bytes, then three publications, `demo_1d.npz`, `demo_1d.png`, `demo_1d.mp4`. Only this catches an output moved into place before a later one has been made.
  - `test_a_failed_clip_leaves_every_existing_output_as_it_was`: for `demo_1d`, an existing set, `find_encoder` returning `"unused"`, and `clocks._demos._common.render_clip` replaced by a stand-in that writes `b"partial"` to the path it is given and raises `EncoderError("the encoder failed")`: `main([..., "--video"])` raises `SystemExit` with code 1; standard error ends `error: the encoder failed`; the three files hold their old bytes; and the directory holds no name starting `.staged-`. Only this catches a new archive and PNG published beside an old clip, or a traceback for a failed encode (Review Focus 2).
  - `test_a_failed_png_leaves_every_existing_output_as_it_was`: the same without `--video`, with `clocks._demos._common.save_png` replaced by a stand-in that writes `b"partial"` to its path and raises `OSError("disk")`: the `OSError` propagates, the three files hold their old bytes, and no staged name is left. Only this catches a PNG written onto its final name, which a failed save would leave cut short.
  - `test_a_run_without_video_keeps_an_earlier_clip_and_says_so`: for `demo_1d`, an existing set, no `--video`: the archive loads and the PNG starts with the PNG signature (both replaced); `demo_1d.mp4` still holds `b"old clip"`; standard output has exactly one line starting `Kept: ` and it names that file. With no clip there beforehand, no line starts `Kept: `. Only this catches a stale clip removed without being asked, or kept without a word (Review Focus 3).
  - `test_a_symlinked_output_is_replaced_not_written_through`: `demo_1d.png` is a symbolic link to `elsewhere.png`, which holds `b"target"` (the test is skipped where `os.symlink` is refused); after the run `demo_1d.png` is a regular file starting with the PNG signature and `elsewhere.png` still holds `b"target"`. Only this catches an output written through a link.
  - `test_video_writes_the_clip` (each clip demo): `--video --observations 2 --particles 16` leaves exactly the three files, and `probe` of the clip reports 92 frames of 1500 by twice the declared height. Only this catches `--video` writing nothing, or a clip of another run's length.
  - `test_density_has_no_video_option`: `main(["--video", "--observations", "1", "--particles", "16"])` for `demo_density` raises `SystemExit` with code 2 and writes nothing.
  - `test_video_without_an_encoder_stops_before_the_run` (each clip demo): with `clocks._demos._common.find_encoder` raising `EncoderNotFound` (the library's own message) and the module's `record` replaced by `pytest.fail`, `main(["--output-dir", "out", "--video", "--observations", "1", "--particles", "16"])` raises `SystemExit` with code 2; standard error contains `imageio-ffmpeg` and `ffmpeg on PATH`; `tmp_path / "out"` does not exist. Only this catches a minute of inference before the command says it cannot make the clip.
  - `test_command_prints_the_recorded_run` (the five clip demos with a filter archive): the module's `record` replaced by a stub that notes its keywords and returns `small(name)` with `true_rates[0]` set to 0.5 and 0.25 times the first prior width added to `mean[-1, 0]`; `main` with `--observations 7 --particles 23` calls it once with those two values; standard output holds the line `f"True rates: {edited}"`, the `Truth:` line built in the test from the stub's `true_params`, and after `Final estimate after 9 observations:` the lines built in the test from the stub's last row with `trace_decimals(...) + 1` decimals and the stub's `ess` and `n_particles`. Only this catches numbers printed from constants or from the arguments.
  - `test_model_comparison_command_prints_from_the_replay_and_records_once`: the same pattern on `demo_model_comparison` with `record` wrapped by a counting spy: one call; the output holds `True model: K=2`, the `MAP model:` line for the candidate of largest final probability, and `Estimate:` and `Std:` lines equal to `str` of that candidate's `final_mean_k<K>` and `final_std_k<K>`; and the module has no attribute `infer`. Only this catches the command running the comparison a second time for its printout.
  - `test_ascii_name`: `x₁` → `x1`, `M₂` → `M2`, `μ` → `mu`, `σ` → `sigma`, `y` → `y`.
  - `test_no_demo_module_imports_the_old_drawing_code`: by syntax tree, no module under `src/clocks/_demos/` imports `clocks.viz`, `clocks._animate`, `clocks._panels` or `clocks._panels3d`. Only this catches a demo that would stop importing when Task 2 deletes those modules.
  - `test_demo_help_exits_without_running`, `test_project_scripts_point_directly_to_packaged_demo_modules`, `test_demo_selects_agg_before_importing_pyplot` and `test_source_demo_is_only_a_thin_wrapper` are kept, with the changed `DEMO_CASES`.

  In `tests/test_density_demo.py`: every `main([...])` passes `--output-dir <dir>` and reads `<dir>/demo_density.png`. `test_density_command_writes_png_bytes_at_the_path_given` is deleted (the command no longer takes a path), and `test_density_command_writes_only_its_png` becomes `test_density_command_writes_exactly_its_archive_and_png`: the directory holds exactly `demo_density.npz` and `demo_density.png`, and the archive's `meta` has the requested counts. Its spies move from `demo_density.save_png` to `clocks._demos._common.save_png`, and the path they expect is the staged one.

  In `tests/test_generated_assets.py`:

  - `test_command_lists_name_the_files_the_commands_write` (IDs `README.md`, `site/reproduce/getting-started.qmd`): for each of the seven commands, the file has a line starting `uv run <executable> ` that contains `output/<demo>.npz` and `output/<demo>.png`; exactly one line starts `uv run demo-2d --video` and it contains `output/demo_2d.mp4`; no line starting `uv run demo-density` contains `--video`; and the file does not contain `.gif`. Only this catches a command list that still promises GIFs, or one that offers `--video` for the density demo.
  - `test_the_commands_are_described_as_they_are` (pull request 3's test of the same name, its sentence replaced): the Reproducibility page contains "Each `uv run demo-*` command records the same configured scenario again, prints its estimates, and writes an archive and a PNG: the final frame for the six animated demos, the density figure for `demo-density`. The six also take `--video`, which writes the clip." and contains neither "still write GIFs" nor "`uv run demo-density` records the same configured scenario and draws its result."
  - `test_readme_describes_the_site_and_the_commands_as_they_are` (pull request 3's test): its assertion of "Six of the commands above still write GIFs, drawn by the earlier animation code; `demo-density` writes a PNG." becomes an assertion that the README does not contain "still write GIFs"; the rest stands.

- [ ] **Step 2: Stage the names, then run.** Add the new functions to `_common.py` (`make_parser` returning a default parser; `run_command` parsing and calling `run` only; `write_outputs` returning `[]`; the printers printing nothing; `ascii_name` returning its argument), keeping the old `add_common_arguments` signature working beside the new one for this step only. Run `uv run pytest tests/test_cli.py tests/test_density_demo.py tests/test_generated_assets.py -q`: the new tests fail on their assertions (GIFs written, `--output` accepted, nothing staged, nothing printed from the replay).

- [ ] **Step 3: Implement.** Imports in each demo module stay after `matplotlib.use("Agg")`: `_common.py` now imports drawing code, and `test_demo_selects_agg_before_importing_pyplot` checks the order. In `README.md` the command block becomes

```bash
uv run demo-1d                  # → output/demo_1d.npz, output/demo_1d.png
uv run demo-2d                  # → output/demo_2d.npz, output/demo_2d.png
uv run demo-multi-mass          # → output/demo_multi_mass.npz, output/demo_multi_mass.png
uv run demo-multi-mass-2d       # → output/demo_multi_mass_2d.npz, output/demo_multi_mass_2d.png
uv run demo-model-comparison    # → output/demo_model_comparison.npz, output/demo_model_comparison.png
uv run demo-density             # → output/demo_density.npz, output/demo_density.png
uv run demo-echolocation-3d     # → output/demo_echolocation_3d.npz, output/demo_echolocation_3d.png
uv run demo-2d --video          # also → output/demo_2d.mp4
```

  with "Six animated demos and one static figure:" replaced by "Seven demos:"; the sentence "Six of the commands above still write GIFs, drawn by the earlier animation code; `demo-density` writes a PNG." deleted; and the paragraph "Most GIF demos animate…" replaced by: "Each command records its run, prints the truth and its final estimate, and writes a replay archive and a PNG: the final frame of the animation for six of them, the density figure for `demo-density`, which wrote a PNG before and now writes its archive beside it. The six animated demos also take `--video`, which writes the animation as an MP4; it needs an FFmpeg encoder, which `uv sync` provides in a checkout. `--theme dark` draws the dark version and `--output-dir` chooses the directory. All seven, with commentary: [jbwhit.github.io/clocks](https://jbwhit.github.io/clocks/). The echolocation range study behind the site's final page: `scripts/scan_echolocation_range.py`." Getting Started gets the same block under "Seven demos. Each writes a replay archive and a PNG; for the six animated ones `--video` adds an MP4:". On the Reproducibility page the two sentences about the commands become the two the test pins.

- [ ] **Step 4: Tests pass; full gate.** Then remove `dist/`, run `uv build`, and run `uv run python scripts/check_wheel_entrypoints.py dist/<the one .whl there> python3`, naming the file, which must end with the two new `ok:` lines (`dist/` is git-ignored). Render the site and run both checks (two pages changed).

- [ ] **Step 5: Commit and push** as `Demo commands record, print from the replay and write an archive and a PNG; --video adds the clip`. The body says that `--output` is gone, that options can no longer be abbreviated, what each command now writes, and what happens to an older clip.

---

### Task 2: The public names, the old modules, the version and the upgrade note

**Files:**
- Delete: `src/clocks/_animate.py`, `src/clocks/_panels.py`, `src/clocks/_panels3d.py`, `tests/test_viz.py`
- Modify: `src/clocks/viz.py`, `pyproject.toml`, `uv.lock`, `README.md`, `site/reproduce/architecture.qmd`
- Test: `tests/test_api.py`, `tests/test_generated_assets.py`

- [ ] **Step 1: Confirm nothing else uses what is being deleted.** The pattern is built from the three module names and all nineteen removed names:

```bash
grep -rnE "_animate|_panels|animate_(echolocation|inference|inference_2d|inference_multi_1d|inference_multi_2d|model_comparison)\b|create_(echolocation|inference)_dashboard|plot_(centered_rates|clock_rates|clock_rates_2d|clock_setup|clock_setup_2d|mass_histogram|particle_cloud|particle_cloud_2d|particle_cloud_multi_1d|particle_cloud_multi_2d|scene_3d)\b" src scripts site tests README.md
```

  It must find only `src/clocks/viz.py`, the three modules themselves, `tests/test_viz.py`, and the README and Architecture lines this task rewrites (K3). `grep -n "^class \|^def test" tests/test_viz.py` lists eighteen classes, each of which exercises only removed names; if one tests something that survives, move it to `tests/test_clips.py` and say so.

- [ ] **Step 2: Write the failing tests.**

  In `tests/test_api.py`:

  - `test_viz_exports_exactly_the_documented_names`: `set(clocks.viz.__all__) == {"THEMES", "Palette", "density_figure", "load_replay", "palette", "range_study_figure", "render_clip", "show", "style"}`, and each is an attribute. Only this catches a name added or dropped without the upgrade note changing.
  - `test_the_removed_names_are_gone`: with `ANIMATE` the six names `animate_echolocation`, `animate_inference`, `animate_inference_2d`, `animate_inference_multi_1d`, `animate_inference_multi_2d`, `animate_model_comparison`; `PLOT` the eleven `plot_centered_rates`, `plot_clock_rates`, `plot_clock_rates_2d`, `plot_clock_setup`, `plot_clock_setup_2d`, `plot_mass_histogram`, `plot_particle_cloud`, `plot_particle_cloud_2d`, `plot_particle_cloud_multi_1d`, `plot_particle_cloud_multi_2d`, `plot_scene_3d`; and `DASHBOARD` the two `create_echolocation_dashboard`, `create_inference_dashboard`: none is an attribute of `clocks.viz`, and no module `clocks._animate`, `clocks._panels` or `clocks._panels3d` can be imported. Beyond the `__all__` test, this checks attributes and importability: a name dropped from `__all__` and still importable passes the first and fails this.
  - `test_the_version_is_0_2_0_in_the_project_and_the_lock`: with `tomllib`, `pyproject.toml`'s `project.version` is `"0.2.0"`, and the entry of `uv.lock`'s `package` list named `clocks` has `version == "0.2.0"`. Only this catches a version bump without `uv lock`, which fails CI's `uv sync --locked` (K2).

  In `tests/test_generated_assets.py`:

  - `test_the_readme_has_the_upgrade_note`: the README contains the heading `## Upgrading from 0.1` and, whitespace-normalised, each of: "Removed from `clocks.viz`: the six `animate_*` functions, the eleven `plot_*` functions and the two `create_*_dashboard` functions."; "Added since 0.1: `load_replay`, `render_clip`, `density_figure`, `range_study_figure`."; "`--output PATH` becomes `--output-dir DIR`"; "Each demo writes a replay archive and a PNG where six of them wrote a GIF; an MP4 is opt-in for those six, with `--video`." The counts six, eleven and two are the lengths of the three lists of the test above.
  - `test_the_architecture_page_names_only_modules_that_exist`: every `_name` in the page's Mermaid block that is written as a node label (the text inside `["..."]`, split on `, `, a trailing `*` or `/` dropped, `_clip_` standing for the `_clip_*.py` files) names a module or directory under `src/clocks/`; the page contains none of `_animate`, `_panels`; and it still says that `_style` comes "with its bundled fonts". Only this catches the diagram drifting from the package again.
  - `test_the_readme_structure_names_only_files_that_exist`: every entry of the README's `src/clocks/` listing (a line of the code block under "Project structure" that begins with four spaces and a file or directory name) exists under `src/clocks/`; the listing contains none of `_animate`, `_panels`.

- [ ] **Step 3: Run** `uv run pytest tests/test_api.py tests/test_generated_assets.py -q`; the new tests fail.

- [ ] **Step 4: Implement.**
  - `git rm` the three modules and `tests/test_viz.py`. `clocks.viz` imports and exports exactly the nine names; its docstring reads "Figures, clips and the shared style (public facade). Implementations live in `clocks._figures` and `clocks._echo_study` (figures drawn from committed data), `clocks._clips` (clips drawn from replays), `clocks._replay` (the archive loader) and `clocks._style`."
  - `pyproject.toml`: `version = "0.2.0"`; then `uv lock`, and confirm `git diff uv.lock` changes the `clocks` version line and nothing else.
  - `README.md`, a section before "Run tests":

```markdown
## Upgrading from 0.1

Version 0.2.0 changes what the demos write and removes the old plotting names.
This note covers everything since 0.1.

- Removed from `clocks.viz`: the six `animate_*` functions, the eleven `plot_*`
  functions and the two `create_*_dashboard` functions. Added since 0.1:
  `load_replay`, `render_clip`, `density_figure`, `range_study_figure`.
- `--output PATH` becomes `--output-dir DIR`; the file names are the demo's, and
  options can no longer be abbreviated.
- Each demo writes a replay archive and a PNG where six of them wrote a GIF; an
  MP4 is opt-in for those six, with `--video`. `demo-density` already wrote a
  PNG and now writes its archive beside it.
```

    and in "Project structure" the `viz.py` line becomes `viz.py         Figure, clip and style facade (_figures.py, _clips.py, _replay.py, _style.py)`, the `_panels3d.py` line goes, and these are added in the listing's order: `_replay.py     Replay archives: record, validate, load, compare`; `_figures.py    Static figures drawn from a replay, and the trace helpers`; `_clips.py      Clips drawn from replays (_clip_*.py hold the six builders)`; `_video.py      Raw frames to H.264 and back, through FFmpeg`; `_site_media.py The site's pre-render step: clips and posters into site/media/`.
  - `site/reproduce/architecture.qmd`: in the Mermaid block the `vis` subgraph's nodes become `viz`, `figures["_figures"]`, `clips["_clips, _clip_*"]`, `video["_video"]`, `sitemedia["_site_media"]` and `figstyle["_style, _fonts/"]`; `replay["_replay"]` joins the `scenarios` subgraph; the eight edges from `viz` and `animate` are replaced by `viz --> figures`, `viz --> clips`, `viz --> echo`, `viz --> replay`, `viz --> figstyle`, `figures --> replay`, `figures --> figstyle`, `clips --> figures`, `clips --> video`, `clips --> replay`, `clips --> figstyle`, `sitemedia --> clips`, `sitemedia --> video`, `sitemedia --> replay`, `sitemedia --> figstyle`, `replay --> api`, `echo --> figstyle`; and `demos --> viz` becomes `demos --> clips`, `demos --> figures`, `demos --> replay`, `demos --> video` and `demos --> figstyle`. Before writing each new edge, confirm it with `grep -n "^from clocks\|^import clocks" <the module>`; for the edges from `demos`, look in `src/clocks/_demos/_common.py` as well as the seven entry modules, since the shared module is where the commands reach the clip and video code; leave out any edge the imports do not show. In "In words", "`viz` is the public facade over private animation and panel modules and the shared figure style (`_style`, with its bundled fonts)" becomes "`viz` is the public facade over the private figure and clip modules (`_figures`, `_clips`), the replay loader (`_replay`) and the shared figure style (`_style`, with its bundled fonts)".

- [ ] **Step 5: Tests pass; full gate; render the site and run both checks; then remove `dist/`, `uv build`, and run the entry-point check on the one wheel it holds** (the package's contents and version changed).

- [ ] **Step 6: Commit and push** as `Remove the GIF animators and panel functions; version 0.2.0 with an upgrade note`.

---

### Task 3: Mutation checks

No files change. **The coordinator runs it, with a mutation harness in worktrees that no reviewer or implementer is using**; mutants may run in several worktrees at once, one at a time in each, and no row is dropped or merged. In each worktree the environment is synced once on the clean tree; for every mutant the harness sets `PYTHONDONTWRITEBYTECODE=1`, deletes every `__pycache__`, confirms `git status --porcelain` is empty, applies the mutant and restores it in a `try`/`finally`, and runs the whole suite, default and slow tests together, with `uv run --no-sync pytest -q -m ''`, each run under the per-mutant time limit of pull request 3's Task 14 (three times the clean baseline's duration, the whole process group killed when it passes). **The expected sets below are the declaration for this pull request's head**; before the first mutant the coordinator records the baseline and checks every name in the table against the collected IDs, correcting the table, with a note, where a name is wrong. Every mutant breaks library code, a script, a page or a configuration file, never a test.

| Task | Mutation | Tests that must fail, and no others |
|---|---|---|
| 1 | `make_parser`: `allow_abbrev=False` removed | `test_output_is_no_longer_an_option` (each demo); `test_options_cannot_be_abbreviated` |
| 1 | `add_common_arguments`: `--output` kept as an alias of `--output-dir` | `test_output_is_no_longer_an_option` (each demo) |
| 1 | `add_common_arguments`: the default of `--observations` is 10 for every demo | `test_demo_parser_defaults` (each demo) |
| 1 | `run_command`: the encoder looked up after `run(args)` returns | `test_video_without_an_encoder_stops_before_the_run` (each clip demo) |
| 1 | `run_command`: `EncoderError` not caught | `test_a_failed_clip_leaves_every_existing_output_as_it_was` |
| 1 | `write_outputs`: each output published as soon as it is staged | `test_outputs_are_staged_before_any_is_published`; `test_a_failed_clip_leaves_every_existing_output_as_it_was`; `test_a_failed_png_leaves_every_existing_output_as_it_was` (the archive is already replaced when the PNG fails) |
| 1 | `write_outputs`: the PNG saved straight to its final name | `test_outputs_are_staged_before_any_is_published`; `test_a_failed_png_leaves_every_existing_output_as_it_was`; `test_a_failed_clip_leaves_every_existing_output_as_it_was`; `test_png_is_the_final_frame` (each clip demo); `test_a_symlinked_output_is_replaced_not_written_through`; `test_density_command_saves_the_figure_it_drew` |
| 1 | `write_outputs`: staged files not removed after a failure | `test_a_failed_clip_leaves_every_existing_output_as_it_was`; `test_a_failed_png_leaves_every_existing_output_as_it_was` |
| 1 | `write_outputs`: an existing clip removed when `--video` is not given | `test_a_run_without_video_keeps_an_earlier_clip_and_says_so` |
| 1 | `write_outputs`: the `Kept:` line not printed | the same |
| 1 | `write_outputs`: the `Kept:` line printed whether or not a clip exists | the same |
| 1 | `write_outputs`: `style("light")` whatever `theme` says | `test_theme_dark_draws_the_dark_png` (each demo) |
| 1 | `write_outputs`: `save_png` given `plt.figure(figsize=fig.get_size_inches())`, a blank figure of the same size | `test_png_is_the_final_frame` (each clip demo); `test_density_command_saves_the_figure_it_drew` |
| 1 | `write_outputs`: the directory not created | `test_demo_main_writes_its_archive_and_png` (each demo) |
| 1 | `demo_1d._run_demo`: `n_particles` not passed to `record` | `test_command_prints_the_recorded_run[demo_1d]`; `test_demo_main_writes_its_archive_and_png[demo_1d]` |
| 1 | `print_estimate`: the ESS line's denominator read from a module constant of 1000 | `test_command_prints_the_recorded_run` (the five) |
| 1 | `print_estimate`: `trace_decimals(...)` without the added decimal | the same five |
| 1 | `demo_model_comparison._run_demo`: `record` called a second time for the printout | `test_model_comparison_command_prints_from_the_replay_and_records_once` |
| 1 | `demo_density`: `add_video_argument` called | `test_density_has_no_video_option`; `test_command_lists_name_the_files_the_commands_write` does not fail (it reads the documents, not the parser) |
| 1 | `demo_2d.py`: `from clocks.viz import animate_inference_2d` put back, unused | `test_no_demo_module_imports_the_old_drawing_code` |
| 1 | `README.md`: the `demo-2d` line's comment put back to `# → output/demo_2d.gif` | `test_command_lists_name_the_files_the_commands_write[README.md]` |
| 1 | `README.md`: a line `uv run demo-density --video` added | the same |
| 1 | the Reproducibility page: "The six also take `--video`, which writes the clip." changed to "`--video` also writes the clip." | `test_the_commands_are_described_as_they_are` |
| 2 | `clocks.viz`: `render_clip` removed from `__all__` | `test_viz_exports_render_clip`; `test_viz_exports_exactly_the_documented_names` |
| 2 | `clocks.viz`: `from clocks._clips import build_clip as animate_inference` added, not in `__all__` | `test_the_removed_names_are_gone` |
| 2 | `pyproject.toml`: the version put back to `0.1.0`, the lock left | `test_the_version_is_0_2_0_in_the_project_and_the_lock` |
| 2 | `README.md`: the upgrade note's `--output` bullet removed | `test_the_readme_has_the_upgrade_note` |
| 2 | the Architecture page: a node `animate["_animate"]` put back | `test_the_architecture_page_names_only_modules_that_exist` |
| 2 | `README.md`: a line `    _panels3d.py   3D plotting primitives` put back in the structure listing | `test_the_readme_structure_names_only_files_that_exist` |

The expected sets are derived from the tests as this plan specifies them; none was run. The wheel entry-point script has no mutant (decision C11).

- [ ] **Step 1: Complete and check the declaration** against the baseline's collected IDs.
- [ ] **Step 2: Run each mutation** and record the exact set of failing test IDs and any change in what is collected.
- [ ] **Step 3: Class every discrepancy, and resolve it**, exactly as pull request 3's Task 14, Step 3 says: an extra failure, a missing predicted failure, a surviving mutant, a collection failure, a timeout or an unrelated crash; the declared set stays beside the observed one, and no test is changed merely so that the two agree.
- [ ] **Step 4: Report** the table of results with each row's class and resolution. The run counts as evidence only when every row is exact or resolved; until then the hand-over's mutation line says "not complete" and lists what is open.

---

### Hand-over

The coordinator opens the pull request as a draft and gives in its description: what each command writes now and what happens to an older clip; that `--output` is gone and options cannot be abbreviated; the upgrade note; the mutation table with its declared and observed sets and, for each discrepancy, its class and resolution (Task 3, Steps 3 and 4). Before the merge, as in pull request 3's hand-over: after review converges and the pull request is marked ready, the coordinator confirms on that head, by full SHA, that each merge-gating run concluded `success`: `CI`'s default and slow steps and the wheel check's two new `ok:` lines; `Site check`; and the three `Replay report` jobs. After the merge, the `CI` and `Deploy site` runs for the merge commit on `main` are confirmed the same way.

## Self-review

**The design's section 7, line by line.** Commands record, print from the replay, and write an archive and a PNG; `--video`; `--output-dir`; `--theme`; no `--video` for density; video opt-in so the default works from a bare wheel (Task 1). The break documented: version 0.2.0 and the upgrade note with the three removals, the option change and what the demos write (Task 2). The wheel check's added run (Task 1). README and Getting Started show the new commands (Task 1); the README's GIF and its paragraph about committed media were replaced in pull request 3. The Architecture page's module list (Task 2). `docs/someday-maybe.md`'s remark about static GIFs was corrected in pull request 3, when it became false.

**What this plan removes of pull request 3's.** The README's sentence that the commands still write GIFs and the Reproducibility page's sentence to the same effect, with the assertions that pinned them (Task 1). Nothing else of pull request 3 is changed: no archive, no clip, no page embed.

**Names.** `make_parser`, `add_common_arguments(parser, *, observations, particles)`, `add_video_argument`, `run_command(parser, run, argv)`, `write_outputs(replay, directory, theme, figure, encoder=None)`, `print_truth`, `print_estimate`, `ascii_name` and `STAGED` are used with these signatures in every demo module and test.
