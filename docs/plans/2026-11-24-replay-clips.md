# Replay Clips Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw the six demo animations when the site is built, as H.264 clips with posters in both themes, from the committed replay archives; switch the pages, the provenance prose and the demo commands to them; lower the 2D demo's true mass to 0.10; and remove the committed GIFs with the code that made them.

**Architecture:** `clocks._video` turns raw frames into an H.264 file and reads one back; it imports no Matplotlib. The `clocks._clip_*` modules build, from a replay and a palette, a figure with fixed axes whose artists are set to one archive row at a time; `clocks._clips.render_clip` walks a replay's shots through the encoder. `clocks._site_media.render_site_media`, called by a three-line Quarto pre-render script, writes twelve clips and twelve posters to the git-ignored `site/media/`; pages embed them as raw-HTML blocks that the page-source guard pins whole and the rendered-site check decodes, and the demo commands record a run, print it from the replay and write an archive and a PNG, with the clip on request.

**Tech Stack:** Python 3.12+, NumPy, Matplotlib 3.10 (declared `>=3.9`), FFmpeg through `imageio-ffmpeg` (a development dependency) or `ffmpeg` on `PATH`, pytest, uv, ruff, Quarto 1.11.9 on the runner (1.9.32 on the planning machine), GitHub Actions.

**Spec:** docs/plans/2026-11-19-replay-and-animation-design.md

**Status:** Draft for review. Not yet reviewed by a second model family.

The spec's item 3 of section 11 is this pull request. Sections 4, 5, 7, 8, 9 and 10, "Testing" (the "Clips" and "Site" parts), the mutation rows for clip frames, trace and cloud mutations, the decode check, the rendered-site check, the source guard, the box margin and prose, "Checks by eye", "Risks", "Not doing" and "Where this design is least sure" all bind it. Read them before any task. Where this plan and the spec disagree the spec wins and the disagreement is a plan bug to report, except for the departures under "Decisions this plan makes", each of which says why. The plans for pull requests 1 and 2 (`docs/plans/2026-11-19-replay-data.md`, `docs/plans/2026-11-23-replay-figures.md`) show the conventions this one follows.

**Terms.** A *row* is one index k of an archive's per-observation arrays: row 0 is the prior, row k the state after k observations, row T the last. A *clip* is the animation of one demo in one theme, an MP4 file; its *poster* is a PNG of its final frame. A *shot* is one drawing of the figure and the number of video frames it is written for. A *themed element* is an element of the rendered HTML whose class list holds `light-content` or `dark-content`. An *embed* is the raw-HTML block of a page's source that holds a clip's two `<video>` elements; its *description* is the paragraph the videos name with `aria-describedby`. The *manifest* is the per-page list of figures in `scripts/check_site_figures.py`. *Retired* files are committed media this track removes.

## Global Constraints

Tooling and conduct:

- Python only through uv: `uv run pytest`, `uv run ruff format .`, `uv run ruff check .`, `uv run python ...`. Never bare `python` or `pip`.
- **Every task runs the same cycle**, and its steps say only what is particular to it: write the tests of Step 1; run them and see them fail for the stated reason; implement; run them and see them pass; run the full local gate (`uv run ruff format --check .`, `uv run ruff check .`, `uv run pytest -q`, `uv run pytest -q -m slow`) and read its output; then, in a separate command, commit with the task's message and `git push origin claude-replay-clips`. Do not open a pull request (the coordinator does). Begin every git step with an explicit `cd` to the worktree.
- **Every task that changes a page, the site configuration or the pre-render step also renders the site and runs both checks**, after the gate and before the commit: `cd site && uv run --frozen quarto render && cd ..`, then `uv run --frozen python scripts/check_site_links.py` and `uv run --frozen python scripts/check_site_figures.py`. State the local `quarto --version` in the commit body; the runner uses 1.11.9 (`.github/workflows/site-check.yml`, `site.yml`).
- Commit messages are passed with a quoted heredoc (`<<'MSG'`) and end with a `Co-Authored-By:` trailer naming the model that wrote the commit.
- Implementers of Tasks 1 to 15 never use `git checkout`, `git restore` or `git stash` to undo anything: a wrong edit is undone by editing. Task 16's mutants are run by the coordinator's harness in a separate worktree.
- **A red step fails on behaviour, not on a missing name.** Where a task's tests import names that do not exist yet, its Step 2 first adds those names with the signatures of its Interfaces block and bodies that do nothing useful (as each task says), so the run shows the assertions failing.
- **The repository is public.** No committed file may contain an absolute local path, a person's name or an email address. This includes probe scripts, contact sheets (their metadata is checked) and anything printed into a commit body. The encoder's path is printed at run time and never committed.
- **A long job runs in the foreground** with `timeout: 600000`; if it needs longer, wait in a bounded foreground loop, `until ! pgrep -f '[p]ytest -q' >/dev/null; do sleep 10; done; tail -20 <log>`, and call again; never end a turn on a background job. The bracket in the pattern is load-bearing: the harness runs each call as `zsh -c "<command>"`, so an unbracketed pattern matches the loop's own shell and never exits.

The design's values, copied; a task that needs one uses exactly this:

- **Encoding** (section 4): 20 frames per second, `libx264`, `yuv420p`, CRF 20, preset `slow`, `+faststart`, no audio. A frame is drawn once and written as many times as its dwell requires.
- **Pacing** (section 5) at 20 frames per second: the prior holds 1.2 s; observations 1 to 3 hold 0.9 s; 4 to 8, 0.5 s; 9 to 20, 0.25 s; later ones 0.1 s; the last 2.5 s. That is 14.8 s for 50 observations and 17.8 s for 80: 296 and 356 frames.
- **Canvas** (sections 4 and 5): 750 CSS pixels wide; declared heights `demo_1d` 450, `demo_2d` 450, `demo_multi_mass` 500, `demo_multi_mass_2d` 550, `demo_model_comparison` 420, `demo_echolocation_3d` 550. A frame and a poster are 1500 pixels by twice the declared height. Source Sans 3 through `style(theme)`; constrained layout off; every axes rectangle and limit computed once from the archive and fixed for the clip.
- **Status line** (section 5): top left, "Before any observation", then "After k observations". Direct labels, no legend boxes.
- **Colour roles** (section 5): particles are `belief` dots of one size and opacity; a second mass is `belief_alt`, drawn with a different marker and labelled; the truth is `truth`; clocks and observed data are `ink`; the head wireframe is `muted`; the model-comparison lines are `ordered[0..2]`. No clip has a threshold, so none has a dashed line. No library drawing code contains a literal colour: every colour is a field of the `Palette` passed in, or a blend of `background` and `ink`.
- **Traces** (section 5): one panel per parameter: mean, a band of one standard deviation, the truth as a solid line, x from 0 to T. The header gives the name, "true 1.50" and "1.42 ± 0.11". The y-limits are those of `clocks._figures.trace_limits`, which leaves the prior row out.
- **Ellipse** (section 5): drawn from `cov` at two standard deviations and labelled "covariance, 2 s.d.".
- **Zoom inset** (section 5), for one-mass clouds: a fixed window centred on the truth, with half-width 1.25 × (final offset from the truth + two final standard deviations) per axis, labelled with its magnification.
- **Camera** (section 5): with F video frames before the final hold, frame f has azimuth −75° + 40° × (1 − cos(π f / (F − 1))) / 2 and elevation 18°; the camera is still during the hold.
- **Renderers never call the clock-rate forward model** (section 1). No module added here imports `clocks.physics`, `clocks.inference`, `clocks.api` or `clocks._scenarios`.
- **Media files** (section 4): `site/media/<demo>.<theme>.mp4` and `site/media/<demo>.<theme>.png`, git-ignored, listed in `_quarto.yml` under `project.resources` as `media/*.mp4` and `media/*.png`, written by `site/prerender.py` named under `project.pre-render`.
- **Markup** (section 4): each `<video>` carries `class`, `aria-label`, `aria-describedby`, `src`, `poster`, `width="750"`, `height`, `controls`, `playsinline`, `preload="none"`; no `autoplay`, no `loop`, no script. `site/styles.css` gains `figure.replay video { max-width: 100%; height: auto; }`.
- **Encoder lookup** (section 4): `imageio_ffmpeg`, then `ffmpeg` on `PATH`, then an error with instructions. The library depends on neither.
- **Commands** (section 7): `uv run demo-2d` writes `output/demo_2d.npz` and `output/demo_2d.png`; `--video` adds `output/demo_2d.mp4`; `--output-dir` replaces `--output`; `--theme` is new; `demo-density` has no `--video`. The version becomes 0.2.0.
- **The 2D truth** (section 8): `TRUE_M` in `demo_2d.py` goes from 0.15 to 0.10. Every shipped continuous truth value lies inside its prior box by at least 5% of the box's width.
- **Nothing computed is compared exactly across machines.** A value computed in floating point is compared at `rtol=1e-12` or looser (with `atol=1e-12` near zero); a literal is compared exactly. **No test hashes or byte-compares a clip, a poster or a figure**, and none asserts a clip's size in bytes or its bit rate ("Not doing"): encoded bytes differ between FFmpeg builds (Q2). Pixels are compared only between two renders made in the same process.
- **The archives other than `demo_2d.npz` are not re-recorded** unless Task 1 hits its stop condition. At the end, `git diff --stat main -- assets/replays` lists `demo_2d.npz` only.

**The clip artists' vocabulary.** Tests find axes and artists by gid, so these names are an interface of every clip task. Axes: `"cloud"` (an (x, M) plane), `"map"` (an (x, y) plane), `"strip"` (the clocks under a cloud panel), `"slowing"` (the per-clock panel), `"scene"` (the 3D close view), `"overview"` (the whole prior box), `"fit"` (observed against fitted), `"posterior"` (model probabilities), `"trace-<name>"` with `<name>` from `meta["param_names"]`, and `"inset"` (or `"inset-1"`, `"inset-2"`), which is a child of its panel and is reached through `ax.child_axes`. Artists, with `-1` and `-2` suffixes where a clip has two masses: `"dots"`, `"truth"`, `"ellipse"`, `"clocks"`, `"trace-mean"`, `"trace-band"`, `"trace-truth"`; the figure-level text `"status"`. Each task adds the names particular to its clip.

## Review Focus

Inputs and failure modes the design implies but spells out no test for, most likely to bite first. Each has a test in the task that owns the code.

1. **The layout engine comes back after the first frame.** Under `style(theme)`, `figure.constrained_layout.use` is on. A figure made with `layout="none"` has no engine, and Matplotlib 3.10.8 restores "no engine" after a save by reading that setting again, so the figure has a `ConstrainedLayoutEngine` from its second frame on and every later save warns (measured, Q3). A figure given a placeholder engine with `fig.set_layout_engine("none")` after creation keeps it. Task 3 (`new_figure`) tests the engine after three frames, under warnings as errors.
2. **A one-observation, one-particle run.** `tests/test_cli.py` runs every command at one observation and sixteen particles, the wheel check will run two observations, and `--particles 1` is accepted. With T = 1 the dwell list is `[24, 50]` and a trace has one row after the prior; with one particle every standard deviation is 0, so the ellipse degenerates to a point and the zoom window, by the design's rule, has zero width wherever the estimate sits on the truth. Tasks 2 to 6.
3. **The encoder is missing, or dies part-way, while an older clip is in place.** A clip must be whole or absent under its final name, an existing file must survive a failed encode, and `--video` with no encoder must stop before inference and write nothing. A failing encoder shows up first as `BrokenPipeError` on a frame write (measured, Q2). Tasks 2 and 14.
4. **A truncated clip decodes without an error.** Cut to half its bytes, a clip keeps its `ftyp` and `moov` boxes, its stream line and exit status 0; only the decoded frame count changes, 129 of 296 (measured, Q2; the design's M10 found 97). The decode check must count frames, and must fail loudly when a build's output cannot be parsed. Tasks 2 and 7.
5. **A clock whose running mean of slowing is negative, and a close-view row with dots on both sides of the cube.** A reading above 1 makes a clock's running mean of 1 − rate negative early in a run; an unclipped blend then gives a colour outside the `background`-to-`ink` segment, or an RGBA value below 0, which Matplotlib rejects. No committed archive has one (measured, Q8), so only a constructed row reaches it. Likewise the small recordings have no row with dots both inside and outside the close view (Q9); the committed prior row has 2 of 500 inside. Tasks 3 and 6.

## Decisions this plan makes

Each is something the spec leaves open, or states in a way that conflicts with the code or with itself. The reason follows, then what it costs if wrong.

**Order and size**

- **D1. The site switches before the commands do.** Section 11 lists the commands before the embeds. In that order the Reproducibility page's sentence that each committed GIF "is the packaged default output of its `uv run demo-*` command" is false from the command task until the switch, and the committed 2D GIF disagrees with its command from the truth change on. Here the clips are built (Tasks 2 to 6), the site switches (7 to 10), the 2D truth changes (11), and only then do the commands and the public names change (14, 15). Every pinned sentence is true at its own commit. Cost if wrong: none to the result; the tasks are the design's.
- **D2. The 2D truth changes after the switch, not before.** The design's section 8 says the change touches "the clip on the landing page and Into the Plane", which exist only after the switch. For one commit (Task 10) the 2D clip and its description show the run at 0.15, honestly; Task 11 re-records and the description test, which reads the archive, forces the numbers to follow. Cost: the 2D description is edited twice.
- **D3. Sixteen tasks, one plan, with a split point after Task 13.** The earliest boundary at which the branch could merge with the site correct and no renderer unused is after Task 10: the clips are embedded and checked, the GIFs are gone and the prose is true. This plan recommends splitting three tasks later, after Task 13, so that the 2D truth, the protocol wording and the checks by eye merge with the site they belong to. At either boundary the commands still write GIFs through `_animate.py`, and the README and Getting Started still say so, truthfully. Tasks 14 and 15 (commands, public names, version, upgrade note, removal of the old modules) and a second mutation run could be a fourth pull request. If the pull request is split there, Task 16's table is split by the task column.
- **D4. Task 1 draws with throwaway code and times on the runner through a temporary workflow.** The design wants every clip drawn and timed, a clean build on Quarto 1.11.9 and cold-runner timings before any layout number is final. The probe script is committed under `docs/plans/2026-11-19-replay-and-animation/` (where pull request 2 put its contact-sheet script) so the runner can run it; `.github/workflows/clip-probe.yml` runs it on a push, and exists only on a throwaway branch that is deleted in the same task: a new workflow cannot be started by hand until it exists on `main`, the gated workflows do not run on a draft, and a fifth file under `.github/workflows/` on this branch would fail the test that ties the Reproducibility page's list to that directory. The probe script itself is deleted in Task 13, when the contact sheets are drawn by the real builders. Cost: the drawing is written twice; the gain is that tests are written against a layout that has been looked at.
- **D5. The format's stop condition is stated, and a second version is a plan amendment, not an improvisation.** See Task 1, Step 7.

**Modules and interfaces**

- **D6. Module layout.** `src/clocks/_video.py` (frames to a file and back, no Matplotlib, so the rendered-site check can import it cheaply); `_clip_parts.py` (the figure, status line, traces, dots, ellipse, inset); `_clip_maps.py` (`demo_2d`, `demo_multi_mass_2d`); `_clip_tracks.py` (`demo_1d`, `demo_multi_mass`); `_clip_models.py` (`demo_model_comparison`); `_clip_echo.py` (`demo_echolocation_3d`); `_clips.py` (the registry, `shots`, `final_frame`, `render_clip`); `_site_media.py` (`render_site_media`). `frame_rgba` joins `save_png` in `_style.py`, sharing its settings. The split keeps each file under the repository's 500-line guideline and puts the two clips that share panels together. Cost: eight small modules where the design names one.
- **D7. `render_clip(replay, path, *, theme="light", encoder=None) -> int`** is the public name section 7 lists; the design gives no signature. It returns the number of frames written. `Clip` and `Shot` are private.
- **D8. Clouds are marker-only lines, not scatter collections.** `Line2D.get_data()` and `Line3D.get_data_3d()` are public and return the plotted coordinates; a 3D scatter has no public getter for its points. The design asks for dots "of one size and opacity", which a line's markers are. Cost: no per-dot size, which no clip needs.
- **D9. The ellipse is a closed 121-point line**, `mean + 2·(columns of the covariance's eigen-decomposition)·(cos t, sin t)`. A test can then state the property itself: every vertex is at Mahalanobis distance 2 from the mean under the archive's covariance block. An `Ellipse` patch would be tested through its width, height and angle, which a wrong but self-consistent decomposition satisfies.
- **D10. The decode check parses FFmpeg's own report.** `imageio-ffmpeg` ships no `ffprobe` (design, M10). `probe` runs the encoder as a decoder with `-f null - -progress pipe:1 -nostats`, reads the codec, pixel format, size and frame rate from the input's `Video:` line and the frame count from the last `frame=` line of the progress output, and reads the box order from the file itself. Two builds give the same line here (Q2). A line the pattern does not match raises with the text, so a differently worded build fails the check instead of passing it. Cost: a future FFmpeg that rewords the line breaks the check loudly until the pattern is widened.
- **D11. A test that needs the encoder fails when there is none; it is not skipped.** The design's "Testing" says so. `uv sync` installs the development group, which CI and both site workflows use, so the encoder is present there. A contributor who installs only the wheel and runs the suite sees failures that name the encoder. If the Linux wheel's binary does not run (the design's untested risk), Task 1 finds out on the runner before anything depends on it.
- **D12. Frames go to the encoder as RGBA**, the bytes `savefig(format="rgba")` writes, with `-pix_fmt rgba` on the input: no conversion in Python. They are the pixels `save_png` writes (measured equal, Q3).
- **D13. Every clip and poster is written to a temporary name beside its destination and moved into place** (`os.replace`), so a failure leaves the old file or none. The archive a command writes is already atomic (`save_replay`). The PNG a command writes goes through `save_png` in place, as `demo-density` does today.
- **D14. A full render deletes only media.** The design says a full render "deletes anything else in `site/media/`". `render_site_media` deletes regular files directly inside the media directory whose suffix is `.mp4`, `.png` or `.tmp` and which it did not just write; it never recurses and never touches another suffix. Only `media/*.mp4` and `media/*.png` reach the site, so nothing stale can be published; a wrong path costs at most media files. Cost: a stray file of another type stays in a git-ignored directory.
- **D15. A clip is rebuilt unless both of its files are strictly newer than its archive and every source file.** The design says "older than"; on a file system with coarse timestamps an edit and a render can share one, and rebuilding on a tie is the safe side.
- **D16. Model-comparison lines are coloured by position in `model_ks`**, and a replay with more than three candidates is refused with `ValueError`: `ordered` has three colours. The design's literal fixture has two candidates and must draw.

**Drawing details the design leaves open**

- **D17. "After 1 observation" is singular.** The design writes "After k observations".
- **D18. The zoom label** reads `zoom ×12` when both axes' magnifications round to the same whole number and `zoom ×12 (x), ×9 (M)` otherwise; a per-axis window in an inset of fixed shape has two magnifications. A window whose half-width would be zero (a one-particle run on the truth) takes 1% of that parameter's prior range instead.
- **D19. The slowing panel's limits hold the prior row.** The design leaves the prior out of the *trace* limits and says the slowing panel's `belief` bars "show the prior's prediction" before any observation, so they must be in view. Task 1 may reverse this if the prior's bars flatten the rest; the test asserts fixity and containment, not the numbers.
- **D20. Layout rectangles are data, fixed by Task 1.** Each clip module holds one `LAYOUT` mapping of axes gid to a rectangle in figure fractions. Tests assert that the rectangles lie on the canvas, do not overlap and never move; they do not pin the numbers, which are a matter for the eye.
- **D21. Descriptions use the trace headers' decimals** (`clocks._figures.trace_decimals`: two for positions, three for masses), so a description, a header and a poster agree digit for digit. The commands print one more decimal than that (three for positions, as today, and four for masses), because review of a re-recording rests on the printed estimates (design, "Risks"). `demo-density`'s printing is left as pull request 2 pinned it.

**The site**

- **D22. An embed is a fenced raw block**, ```` ```{=html} ````, not bare HTML: Pandoc then passes it through byte for byte, and the guard can pin the whole block as one string built from a table. Task 1 confirms on Quarto 1.11.9 that the block and its attributes arrive unchanged. The description is a fenced div, `::: {#demo-2d-text}`, directly after it.
- **D23. The source guard pins each embed whole and bans video markup anywhere else on a page.** That is the lesson of pull request 2: a rule list drew one more spelling each round. The guard's limit is written in the test file: it catches an embed changed by mistake; it does not prove that a clip is visible or plays on the rendered page, which is the rendered-site check's and the browser checks' job.
- **D24. The landing page and "Into the Plane" carry the same description** of the 2D clip, word for word. Seven embeds, six descriptions.
- **D25. The stylesheet rule is required only when the manifest lists a clip**, so the checker's plot-only fixtures need no stylesheet.
- **D26. `scripts/check_site_figures.py` imports `clocks`** (`clocks._video`, `clocks._replay`) for the first time. It already runs under `uv run`, where the package is installed.
- **D27. The check reads each clip's expected frame count from its archive** (`assets/replays/<clip>.npz`, `n_observations`), so the count cannot go stale in a second table.

**Commands and names**

- **D28. `--video` with no encoder stops before inference**, through `parser.error`: exit status 2, a message naming `imageio-ffmpeg` and `ffmpeg`, nothing written. The design says the command "stops and says how to get one" without saying when.
- **D29. What each command writes**: `<dir>/<demo>.npz` (replaced whole), `<dir>/<demo>.png` (overwritten), and with `--video` `<dir>/<demo>.mp4` (replaced whole), in that order, `<dir>` created if missing, default `output`. `--output` is removed with no alias and is an argparse error, as the design's "replaces" says.
- **D30. `demo-density` writes its archive too**, with `--output-dir` and `--theme` and no `--video`. Section 7 says each command "records its run" and writes an archive and a PNG, and names the density figure as that command's PNG. The pull-request-2 tests that it "writes only its PNG" become tests that it writes exactly those two files.
- **D31. The old names go at once**, with no deprecation period: section 7 calls it a documented break. `clocks.viz` loses the six `animate_*`, eleven `plot_*` and two `create_*_dashboard` names; `render_clip` is added in Task 3, when it first exists (adding a name breaks nothing).
- **D32. The README's GIF and its paragraph go in Task 10**, with the file they show, not with the commands: a README image of a deleted file is a broken image, and the retired-asset guard forbids the mention.

**Prose and records**

- **D33. "Run exactly once" becomes a protocol statement in five places**: the four the design names and `docs/someday-maybe.md`, which repeats the sentence. `scripts/scan_echolocation_range.py` keeps its notice, which is an instruction printed when a certification run starts, not a claim about the past.
- **D34. The earlier seed blocks are disclosed, with citations checked by the implementer.** The owner's decision is "reworded as a protocol statement, with the earlier seed blocks disclosed"; the design allows the sentence "only with a citation the implementer has checked". The citations exist in the repository (read while planning): seeds 100–111 used as a holdout and then set aside (`docs/superpowers/specs/2026-07-03-clone-freeze-diagnosis.md`), seeds 200–211 as a certification block (`docs/superpowers/specs/2026-07-02-annealed-jitter-design.md`), seeds 300–311 as the echolocation certification block of 2026-07-19 (`docs/superpowers/specs/2026-07-19-3d-echolocation-design.md`).
- **D35. The calibration report keeps its six GIF rows and gains one dated sentence per file**, as pull request 2 did for the two PNGs.
- **D36. The audit's status table** gains rows for C6, C9, V1 and V5 and the "still unguarded" note on T2 is brought up to date; the audit's body is a record and is not edited.
- **D37. Deferred items from pull request 2's review.** `gh pr view 28 --comments` lists two explicitly, both Minor and both in `tests/test_site_figure_style.py`; Task 9 takes them. From the coordinator's list this plan also takes, because their files are edited here: a test that `site_libs/` is skipped and a test of the unknown-kind `ValueError` (both Task 7). Left: label positions in the two static figures, which stay unpinned. They are layout, checked on the contact sheets, and `_figures.py` and `_echo_study.py` are not edited here.

## Measurements

Run on 2026-11-24 in the worktree at `177a425`, macOS arm64, Python 3.14.0, NumPy 2.4.2, Matplotlib 3.10.8, through `uv run`. Probe scripts were in a session scratch directory and are not kept. A number in this plan is one of these, or is copied from the design with its section, or is marked believed.

- **Q1. Encoders here.** `uv run --with imageio-ffmpeg python -c "import imageio_ffmpeg as m; print(m.__version__, m.get_ffmpeg_exe())"` resolved 0.8.1 under the project's seven-day cooldown; its binary is `ffmpeg-macos-aarch64-v9.0`, FFmpeg 9.0 built with `--enable-gpl --enable-libx264`. Homebrew's `ffmpeg` on `PATH` is 9.0.2. The Linux binary was not run.
- **Q2. Encoding and decoding.** 296 synthetic frames of 1500 by 900, piped as raw frames with the settings of Global Constraints, through each build: exit 0, 27,454 bytes each, **different bytes** between the two builds; boxes `ftyp`, `moov`, `free`, `mdat`. Decoding either file with either build (`-hide_banner -nostdin -i <file> -map 0:v:0 -f null - -progress pipe:1 -nostats`): the same line, `Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1500x900, 12 kb/s, 20 fps, 20 tbr, 10240 tbn (default)`, and a last progress line `frame=296`. The decoder's standard error holds two `Video:` lines, the input's first and then one for its null output. The file cut to half its bytes: exit 0, the same boxes and stream line, `frame=129`, with `Invalid NAL unit size` on standard error. An output path in a directory that does not exist: the encoder exits and the next frame write raises `BrokenPipeError`. Small clips, both builds: three solid frames of 64 by 48 written 2, 1 and 4 times decode to `frame=7` with the same boxes; copied with `-c copy` and no flag the boxes are `ftyp`, `free`, `mdat`, `moov` and the count is still 7; sixty frames of seeded noise make 117,153 bytes, and the half file decodes to `frame=29` with its stream line intact; a text file named `.mp4` gives exit 183 and no `Video:` line; re-encoded with `-c:v mpeg4`, and with `-c:v libx264 -pix_fmt yuv444p`, the lines are those quoted in Task 2's parser test. `-version` prints `ffmpeg version ...` as the first line of standard output.
- **Q3. Frames.** Inside `style(theme)`: `plt.figure(figsize=(7.5, 4.5), layout="none")` reports no layout engine; after one `savefig` it reports a `ConstrainedLayoutEngine`, and a second save after `add_axes` warns "There are no gridspecs with layoutgrids". `plt.figure(figsize=(7.5, 4.5))` followed by `fig.set_layout_engine("none")` reports a `PlaceHolderLayoutEngine`, the same object after four saves and a draw, with no warning under `warnings.simplefilter("error")` and with a plain axes, an inset and a 3D axes all keeping their positions. `fig.savefig(buffer, format="rgba", dpi=200)` gives 1500 by 840, 900, 1000, 1080 and 1100 pixels for heights of 4.2, 4.5, 5.0, 5.4 and 5.5 inches, every alpha 255, equal pixel for pixel to the file `save_png` writes. A 500-dot figure with two axes redraws and saves in about 20 ms; a 3D axes with 500 dots in about 32 ms. `ax.inset_axes(...)` is in `ax.child_axes` and not in `fig.axes`. A styled canvas with nothing on it has 1 distinct colour; with one empty axes about 218; with a status line about 447.
- **Q4. The dwell schedule**, by the rule of Global Constraints: 296 frames for T = 50 and 356 for T = 80; `[24, 50]` for T = 1; `[24, 18, 50]` for T = 2.
- **Q5. The camera**, by the design's formula with F = 306 (T = 80): frames 0, 76, 152, 229 and 305 have azimuths −75.0, −69.17850576840875, −55.103002582480016, −40.82149423159126 and −35.0. A constant-speed turn would give −65.03 at frame 76. With F = 24 (T = 1): frames 0, 6, 12 and 23 give −75.0, −68.65106286437309, −53.63515173270658 and −35.0.
- **Q6. The 2D run at M = 0.10**, in a scratch clone with `TRUE_M = 0.10` and `scripts/record_replays.py --out assets/replays --overwrite demo_2d`: final mean (1.41692, −1.22229, 0.10002) ± (0.11108, 0.11479, 0.00124), the design's M2; after observation 1, (3.630, −3.361, 0.1225) ± (1.465, 1.502, 0.0152); after 8, (1.861, −1.352, 0.0993) ± (0.304, 0.298, 0.0032). Largest |x–y correlation| after the prior: row 3, −0.690. Running mean of slowing: largest over the run 0.031689, largest in the last row 0.024270, 2 of 400 clock-rows above 1.1 times the latter. Zoom half-widths by the design's rule: 0.38156 (x), 0.56483 (y), 0.0031185 (M). Weak-field strength 0.0468. With that archive, `uv run pytest -q -m '' tests/test_replays.py tests/test_demo_records.py tests/test_shipped_ground_truths.py tests/test_record_replays_script.py`: 237 passed. The "Into the Plane" cell at 0.10: (1.3716, −1.0980, 0.1003) ± (0.1724, 0.1696, 0.0018), largest particle mass 0.1068, 0.162 from the true position; today, at 0.15: (1.4049, −1.0768, 0.1487) ± (0.1114, 0.1140, 0.0010), largest particle mass 0.1500.
- **Q7. `TRUE_M` put back to 0.15 against the 0.10 archive**, whole suite (`uv run pytest -q -m ''`): 3 failed, 1,918 passed. The three: `tests/test_replays.py::test_archive_matches_its_scenario[demo_2d]`, `::test_gate_inputs[demo_2d]`, `::test_gate_summaries[demo_2d]`.
- **Q8. The committed archives**, from `load_replay`. Final mean ± standard deviation at the trace headers' decimals: `demo_1d` (T 50, 1,000 particles, 3 clocks) x 2.48 ± 0.04, M 0.101 ± 0.001; `demo_2d` at 0.15 (T 50, 2,000, 8 clocks) x 1.43 ± 0.07, y −1.15 ± 0.08, M 0.149 ± 0.001; `demo_multi_mass` (T 80, 3,000, 5 clocks) x₁ −2.00 ± 0.04, x₂ 4.55 ± 0.05, M₁ 0.045 ± 0.001, M₂ 0.030 ± 0.001, and after observation 1 x₂ 6.61 ± 1.04, after 8 x₂ 4.62 ± 0.18; `demo_multi_mass_2d` (T 80, 4,000, 10 clocks) x₁ −2.38 ± 0.42, y₁ 2.03 ± 0.22, x₂ 4.13 ± 0.80, y₂ −0.95 ± 0.63, M₁ 0.052 ± 0.004, M₂ 0.028 ± 0.003; `demo_echolocation_3d` (T 80, 6,000, 27 clocks) x 0.98 ± 0.02, y 1.46 ± 0.02, z 2.93 ± 0.03, M 0.078 ± 0.002, `view_bounds` (−1.3, 4.8), `weight_in_view[0]` 0.0045 (printed to one decimal as 0.4%), smallest later value 0.9999999999999998, dots inside the view in rows 0, 1 and 80: 2, 500 and 500. `demo_model_comparison` (T 80, 2,000 per candidate): `model_ks` 1, 2, 3; prior one third each; the leader is K = 3 from observation 1, K = 2 from 10, K = 3 from 19; final (0, 0.39904, 0.60096). Position standard deviations after observation 1 against the prior's: `demo_1d` 0.27 against 4.92; echolocation 0.16 to 0.26 against 9.2 to 9.3. The running mean of slowing is positive in every clock-row of `demo_1d`, `demo_2d` and `demo_multi_mass` (smallest 0.0102); for `demo_2d` at 0.15 its largest value over the run is 0.042842 and its largest in the last row 0.036333, a ratio of 0.848.
- **Q9. Small recordings.** `record(n_observations=9, n_particles=40)` and `record(n_observations=1, n_particles=16)` succeed for all six clip demos under warnings as errors, in 0.02 to 0.18 s each. Echolocation at 9 and 40: `view_bounds` (−1.3, 4.6); dots inside the view per row: 0, then 40 in each later row.
- **Q10. The suite today.** 1,921 tests: 1,913 in the default run, 8 slow. `uv run pytest -q -m ''` took 3 min 19 s.
- **Q11. The project's version is in the lock.** `uv.lock` holds `name = "clocks"`, `version = "0.1.0"`, `source = { editable = "." }`; CI installs with `uv sync --locked`.
- **Q12. What the tree says about GIFs and the old names** (`grep` over the tree outside `docs/plans`, `docs/audits`, `docs/superpowers`, `.venv` and build output). GIF files: six under `assets/`, the same six under `site/assets/`. Embeds: `site/index.qmd` line 24; `site/story/the-search-in-one-dimension.qmd` 11; `into-the-plane.qmd` 23; `two-hidden-masses.qmd` 11 and 13; `how-many-masses.qmd` 15; `gravitational-echolocation.qmd` 108; `README.md` 64. Prose: `README.md` 55 to 61 and 66 to 81; `site/reproduce/getting-started.qmd` 27 to 36; `site/reproduce/reproducibility.qmd` 41, 61 and 87 to 97; `docs/someday-maybe.md` 43; `scripts/check_site_figures.py` 17. Code: the six `OUTPUT_PATH` constants and `demo_model_comparison.py` line 203; `src/clocks/_animate.py`. Tests: `tests/test_generated_assets.py` (the hash table and two sentences), `tests/test_cli.py` (`DEMO_CASES`), `tests/test_viz.py`, and two fixtures in `tests/test_site_figure_style.py` that use `demo_2d.gif` as a name that is not retired. The old `clocks.viz` names are used only by the six demo modules, `tests/test_viz.py` and `clocks.viz` itself; `_animate`, `_panels` and `_panels3d` are also named in `README.md` (lines 108 and 111) and `site/reproduce/architecture.qmd` (lines 29 to 31, 62 to 69 and 89). "Exactly once": `README.md` 88; `site/reproduce/getting-started.qmd` 49; `site/reproduce/reproducibility.qmd` 119; `site/method/the-particle-filter.qmd` 179; `docs/someday-maybe.md` 14; `scripts/scan_echolocation_range.py` 213. No page has a `{=html}` block, a `<figure>` or a `<video>` today.

Not measured, and where each is settled: Quarto 1.11.9, the Linux encoder binary and runner timings (Task 1); the real clips' draw times and sizes (Task 1; the design's M9 has 2.1 s per theme for the 2D prototype and 12.5 s for echolocation); Chrome, Firefox and Safari playback (Task 13).

---

### Task 1: Draw and time every clip; build on Quarto 1.11.9; time the runner; settle the format

Nothing here is library code. The task finds out, before any test pins a layout, whether the six clips can be drawn from the archives as they are, how long they take, and whether Quarto 1.11.9 and the Linux encoder behave as the design measured on Quarto 1.9.32 and macOS.

**Files:**
- Create: `docs/plans/2026-11-19-replay-and-animation/clip_probe.py`; six sheets `docs/plans/2026-11-19-replay-and-animation/clip-<demo>.jpg`
- Modify: `docs/plans/2026-11-24-replay-clips.md` (a "Probe results" section, added at the end)
- On a throwaway branch only, never on `claude-replay-clips`: `.github/workflows/clip-probe.yml`

**Interfaces:**
- Consumes: `load_replay` from `clocks._replay`; `style`, `THEMES` from `clocks._style`; `trace_limits`, `trace_header` from `clocks._figures`; the six committed archives.
- Produces, in `clip_probe.py`: `LAYOUTS: dict[str, dict[str, tuple[float, float, float, float]]]`, for each of the six demos its axes gids (the vocabulary of Global Constraints) mapped to rectangles in figure fractions (an inset's rectangle is in fractions of its parent axes). Tasks 3 to 6 copy `LAYOUTS[<demo>]` into the library. The script takes one of `--sheets DIR`, `--time DIR`, `--quarto DIR`; it imports nothing from `clocks` but the names above, holds no absolute path, and passes `ruff format --check` and `ruff check`.

- [ ] **Step 1: Write the probe's drawing.** One throwaway function per demo, `draw_<demo>(replay, colors, row, azimuth) -> Figure`, following section 5 of the design and the artist paragraphs of Tasks 3 to 6 below, on a figure of 7.5 inches by the demo's declared height made with `plt.figure(figsize=...)` followed by `fig.set_layout_engine("none")` (Q3). Rectangles come from `LAYOUTS`; start from the design's arrangement (cloud or map at left, traces at right; for the two track clips the clock strip and the slowing panel under the cloud; for echolocation the whole-prior box at upper left, the scene at left, traces stacked at right and the fit panel as a small square).

- [ ] **Step 2: Draw the sheets and look at them.** `uv run python docs/plans/2026-11-19-replay-and-animation/clip_probe.py --sheets docs/plans/2026-11-19-replay-and-animation` writes one JPEG per demo: rows k = 0, 1, 8 and the last, in both themes, each at 1500 pixels wide and resampled to 780 (390 CSS pixels at twice density), composed with Pillow and saved with `image.save(path, quality=85)`, no `exif` and no `comment`. For each sheet, report each item as seen or not seen, and change `LAYOUTS` and redraw until all are seen: no text overlaps text or a data mark at full width; the status line and every trace header can be read at 780 pixels, the six headers of the two-mass 2D clip included; no inset covers its truth or the final cloud; `belief` and `belief_alt` clouds can be told apart by marker in both themes; the echolocation ring is open and the final cloud is visible inside it; the whole-prior box is full at k = 0 and holds one cluster at k = 1; the model-comparison labels never stack; nothing in any frame is drawn in a colour outside the palette (the 3D panes and grid included).

- [ ] **Step 3: Time every clip here.** `uv run --with imageio-ffmpeg==0.8.1 python docs/plans/2026-11-19-replay-and-animation/clip_probe.py --time <a directory outside the repository>` renders each of the six clips in both themes, every frame of the dwell schedule, through the encoder with the settings of Global Constraints, and prints one line per clip: demo, theme, drawings, seconds, bytes, and the decoded frame count. Expected frame counts: 296 for `demo_1d` and `demo_2d`, 356 for the other four.

- [ ] **Step 4: Build a scratch Quarto project.** `clip_probe.py --quarto DIR` writes, into an empty directory outside the repository, a project with the site's `custom.scss`, `custom-dark.scss` and `styles.css` (copied from `site/`, plus the responsive rule of Global Constraints), a `_quarto.yml` whose `project` has `pre-render: prerender.py` and `resources: ["media/*.mp4", "media/*.png"]`, a `prerender.py` that writes a twenty-frame clip and a poster per theme into `media/` and appends its working directory, `sys.prefix` and `QUARTO_PROJECT_RENDER_ALL` to a log file, and two pages, `index.qmd` and `story/page.qmd`, each holding the block of decision D22 for `demo_2d` and a `::: {#demo-2d-text}` div. It then runs `quarto render` (with the output directory removed first), `quarto render story/page.qmd`, and a third render with a pre-render script that exits 1, and exits non-zero unless all of these hold:
  - (a) the pre-render script ran in the project directory, under the interpreter the command was started with, and saw `QUARTO_PROJECT_RENDER_ALL=1` on the full render and no such value on the single-page render;
  - (b) after the clean full render, `_output/media/` holds the four files;
  - (c) in each rendered page, the two `<video ...></video>` lines, the `<figure class="replay">` line and the `<figcaption>` line of the source block appear unchanged, attribute for attribute;
  - (d) each page has exactly one element with `id="demo-2d-text"`, holding the div's text;
  - (e) the page's `<body>` has the class `quarto-light`, and a stylesheet under `_output/site_libs/` holds a rule hiding `.light-content` under `body.quarto-dark` and one hiding `.dark-content` under `body.quarto-light`;
  - (f) `_output/styles.css` holds the responsive rule;
  - (g) the render whose pre-render script exits 1 exits non-zero.
  Run it with the local Quarto and record its version.

- [ ] **Step 5: Run Steps 3 and 4 on a cold runner with Quarto 1.11.9.** Commit the probe script and the sheets to `claude-replay-clips` and push (the gate first: the script is linted). Then, from the worktree, `git worktree add ../claude-replay-clips-probe -b claude-replay-clips-probe HEAD`; in that second worktree add `.github/workflows/clip-probe.yml` as below, commit and `git push -u origin claude-replay-clips-probe`. The workflow file never enters `claude-replay-clips`: `tests/test_generated_assets.py::test_the_ci_bullets_link_exactly_the_workflow_files` requires the Reproducibility page to link exactly the files under `.github/workflows/`.

```yaml
name: Clip probe (temporary)

on:
  push:
    branches: [claude-replay-clips-probe]

jobs:
  probe:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v6
      - uses: quarto-dev/quarto-actions/setup@v2
        with:
          version: "1.11.9"
      - uses: astral-sh/setup-uv@v8.2.0
        with:
          version: "latest"
      - name: Set up Python 3.12
        run: uv python install 3.12
      - name: Install dependencies
        run: uv sync --frozen
      - name: Time every clip
        run: uv run --frozen --with imageio-ffmpeg==0.8.1 python docs/plans/2026-11-19-replay-and-animation/clip_probe.py --time "$RUNNER_TEMP/clips"
      - name: Quarto 1.11.9 clean build
        run: uv run --frozen --with imageio-ffmpeg==0.8.1 python docs/plans/2026-11-19-replay-and-animation/clip_probe.py --quarto "$RUNNER_TEMP/quarto-probe"
```

  Find the run with `gh run list --workflow clip-probe.yml --branch claude-replay-clips-probe --json databaseId,headSha,status,conclusion` (the full 40-character SHA when filtering by commit), wait for a non-empty `conclusion`, and read the two steps' output with `gh run view <id> --log`. Then `git push origin --delete claude-replay-clips-probe`, `git worktree remove ../claude-replay-clips-probe` and `git branch -D claude-replay-clips-probe`. The run's log stays reachable by its URL.

- [ ] **Step 6: Record the results in this plan.** Add a section "Probe results" at the end of this plan, with: the local and runner Quarto versions and items (a) to (g) for each; the encoder's version line on the runner; per clip and theme, drawings, seconds and bytes, here and on the runner, and the two totals; the final `LAYOUTS`, by reference to the script; every change made to decisions D17 to D20; and the run's URL. The same numbers go to the coordinator for the pull request's description. Check each sheet's metadata as pull request 2 did: `Image.open(path).getexif()` is empty, `set(Image.open(path).info) == {"jfif", "jfif_density", "jfif_unit", "jfif_version"}`, and `[marker for marker, _ in Image.open(path).applist] == ["APP0"]`.

- [ ] **Step 7: Decide whether to go on.** Stop, change nothing further, and report to the coordinator if any of these holds:
  - **The format.** A panel of section 5 cannot be drawn from the keys `validate_replay` accepts without importing `clocks.physics` or `clocks.inference` or running a demo again; or a sheet shows that a panel misleads for want of data (for example, the 500 dots cannot show the two-mass 2D cloud's second mode, or an inset shows float32 steps). Planning found every panel's inputs present: the slowing panel reads `predicted_mean`, `predicted_std`, `observations`, `true_rates` and `noise_std`; the 2D clock fills read `observations`; the model lines read `model_posterior`, `model_ks` and `true_k`; the echolocation panels read `view_bounds`, `weight_in_view`, `predicted_mean` and the centred `observations`. If the stop condition holds all the same, the design's rule applies: the format version is raised and the archives re-recorded in this pull request. That is a plan amendment made by the coordinator and reviewed before any later task starts: a Task 1b that sets `VERSION = 2` in `src/clocks/_replay.py`, adds the missing keys to `_array_specs`, to the design's section 1 table and to the `record()` of each demo that needs them, updates the literal fixtures and `test_module_constants` in `tests/test_replay.py`, re-records all seven archives with `uv run python scripts/record_replays.py --out assets/replays --overwrite`, and replaces the Global Constraint that only `demo_2d.npz` changes. A version-1 file is then refused by the loader's existing version check.
  - **Time.** The twelve clips take more than 240 s in total on the runner. The design expects under two minutes; the site jobs have a ten-minute limit and render in 43 to 61 s today (design, M13).
  - **Quarto.** Any of (a) to (g) fails on 1.11.9.
  - **The encoder.** `imageio-ffmpeg`'s Linux binary does not run on the runner, or has no `libx264`.

  A layout that fails Step 2's list is not a stop condition: change `LAYOUTS`, redraw, and say what moved.

- [ ] **Step 8: Commit and push** the plan's "Probe results" (the script and sheets were pushed in Step 5) as `Record the clip probe: sheets, timings, Quarto 1.11.9 and the runner`.

---

### Task 2: `_video.py`: the dwell schedule, the encoder, the decode check

**Files:**
- Create: `src/clocks/_video.py`, `tests/test_video.py`
- Modify: `src/clocks/_style.py`, `pyproject.toml`, `uv.lock`
- Test: `tests/test_video.py`, `tests/test_style.py`

**Interfaces:**
- Produces, in `src/clocks/_video.py`, which imports only the standard library at module level:

```python
FPS = 20

class EncoderNotFound(RuntimeError): ...
class EncoderError(RuntimeError): ...

class StreamInfo(NamedTuple):
    codec: str        # "h264"
    pix_fmt: str      # "yuv420p"
    width: int
    height: int
    fps: float
    frames: int       # frames the encoder decoded, not a count read from the container
    moov_first: bool  # the moov box comes before mdat

def dwell_frames(n_observations: int) -> list[int]: ...
def find_encoder() -> str: ...
def encoder_version(encoder: str) -> str: ...
def encode(frames: Iterable[tuple[bytes, int]], path: Path, *, width: int, height: int,
           encoder: str | None = None) -> int: ...
def parse_stream_line(text: str) -> tuple[str, str, int, int, float]: ...
def box_order(path: Path) -> list[str]: ...
def probe(path: Path, encoder: str | None = None) -> StreamInfo: ...
```

  - `dwell_frames(T)` returns T + 1 counts, one per row: 24 for row 0; for 0 < k < T, 18 when k ≤ 3, 10 when k ≤ 8, 5 when k ≤ 20, else 2; 50 for row T. `ValueError` for T < 1.
  - `find_encoder()` returns `imageio_ffmpeg.get_ffmpeg_exe()` when that import and call succeed (an `ImportError`, or the `RuntimeError` the package raises when it has no binary, means "not this way"); else `shutil.which("ffmpeg")`; else raises `EncoderNotFound("no FFmpeg encoder found: install imageio-ffmpeg (in a checkout, `uv sync` provides it) or put ffmpeg on PATH")`.
  - `encode` pipes each frame, `width * height * 4` bytes of RGBA, to the encoder as many times as its count, and returns the total written. It looks for the encoder (when none is passed) before asking the iterable for anything. What it writes: a temporary file `.<name>.<random>.tmp` beside `path` (the directory must exist), moved over `path` with `os.replace` only after the encoder has exited with status 0. On any failure the temporary file is removed, `path` is left exactly as it was (an older file stays, a missing one stays missing), and the error is `ValueError` for an odd `width` or `height` (raised before the encoder starts) or a frame of the wrong length, the caller's own exception if the iterable raised, and otherwise `EncoderError` naming the encoder, its exit status, `path` and the last lines of its standard error. A `BrokenPipeError` on a frame write means the encoder has exited and is reported as that `EncoderError`. The encoder's standard error goes to a temporary file, not a pipe, so a failing encoder that prints a great deal cannot block the frame writes (this is not tested: a blocked pipe would hang the suite, not fail it). The command is exactly: `[encoder, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", "20", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart", "-f", "mp4", str(temporary)]`.
  - `probe` runs `[encoder, "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1", "-nostats"]`, gives `parse_stream_line` its standard error, takes `frames` from the last `frame=<n>` line of its standard output, and `moov_first` from `box_order`. `ValueError(f"{path}: the encoder reported no video stream: ...")` when there is no `Video:` line or no `frame=` line.
  - `parse_stream_line(text)` finds the first line containing `Video:` and returns codec, pixel format, width, height and frames per second from it: the word after `Video:`, the word after the first comma that follows, the first `<digits>x<digits>` after that, and the number before ` fps`. `ValueError` quoting the line when any part is missing.
  - `box_order(path)` returns the four-character types of the file's top-level boxes in order, following each box's 32-bit size (a size of 1 means a 64-bit size follows the type; a size of 0, or one that runs past the end of the file, ends the walk after that box is listed).
- Produces, in `src/clocks/_style.py`: `frame_rgba(fig: Figure) -> bytes`, the pixels `save_png` writes, as rows of RGBA with no header. `save_png` and `frame_rgba` call one private function with `save_png`'s present settings, differing only in `format`.
- `pyproject.toml`: `imageio-ffmpeg>=0.8.1` in the `dev` group only (`uv add --dev "imageio-ffmpeg>=0.8.1"`, which also rewrites `uv.lock`).

- [ ] **Step 1: Write the failing tests.** In `tests/test_video.py`. "A small clip" is three frames of 64 by 48 pixels, each `bytes([r, g, b, 255]) * (64 * 48)` for three different colours, with counts 2, 1 and 4, encoded with the real encoder into `tmp_path`.

  - `test_dwell_frames` (IDs by T): 50 → `[24] + [18] * 3 + [10] * 5 + [5] * 12 + [2] * 29 + [50]`; 80 → the same with `[2] * 59`; 1 → `[24, 50]`; 2 → `[24, 18, 50]`; 4 → `[24, 18, 18, 18, 50]`; 9 → `[24, 18, 18, 18, 10, 10, 10, 10, 10, 50]`; 21 → `[24] + [18] * 3 + [10] * 5 + [5] * 12 + [50]`; and `sum(dwell_frames(50)) == 296`, `sum(dwell_frames(80)) == 356`. Only this catches two tiers swapped, a tier boundary off by one (the 4, 9 and 21 cases end exactly on one), or the last row given its tier's count.
  - `test_dwell_frames_needs_an_observation`: `dwell_frames(0)` raises `ValueError` matching `at least one observation`. Only this catches a schedule for T = 0 that returns `[24]` or `[50]`.
  - `test_find_encoder_order` (IDs `package_first`, `path_second`, `package_without_a_binary`, `neither`): with `sys.modules["imageio_ffmpeg"]` replaced by a stand-in whose `get_ffmpeg_exe` returns `"from-package"` and `shutil.which` patched to return `"from-path"`, the result is `"from-package"`; with the module entry set to `None` (so the import raises `ImportError`), `"from-path"`; with a stand-in whose `get_ffmpeg_exe` raises `RuntimeError`, `"from-path"`; with the entry `None` and `which` returning `None`, `EncoderNotFound` whose message contains both `imageio-ffmpeg` and `ffmpeg on PATH`. Only this catches the lookup order reversed, or a package without a binary treated as fatal.
  - `test_video_module_imports_only_the_standard_library`: every module named by an `Import` or `ImportFrom` node of `src/clocks/_video.py` that lies outside every function body (a `try` at module level included) has its first component in `sys.stdlib_module_names`. Only this catches `import imageio_ffmpeg` at module level, which would make `clocks.viz` fail to import from a wheel installed without the encoder.
  - `test_encode_then_probe`: the small clip; `encode` returns 7, `tmp_path` holds exactly `clip.mp4`, and `probe(path) == StreamInfo("h264", "yuv420p", 64, 48, 20.0, 7, True)`. Only this catches a wrong codec, pixel format, rate or `+faststart` flag in the command. No byte of the file is compared.
  - `test_the_encoder_is_given_the_designs_settings`: the encoder is an executable script, written by the test, that records its arguments and succeeds (`printf '%s\n' "$@" > "$0.args"`, then `for last; do :; done`, `cat > "$last"`); after `encode` of the small clip with `width=64, height=48`, the recorded arguments without the last are exactly `["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "64x48", "-r", "20", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart", "-f", "mp4"]`, the last is a path in `tmp_path` ending `.tmp`, and `clip.mp4` holds the 7 × 12,288 bytes the script was sent. Only this catches a changed quality setting (CRF, preset, audio), which no decoded property shows; the command is pinned whole, as page cells are.
  - `test_encode_refuses_odd_sizes` (IDs `width`, `height`): `width=63` or `height=47` raises `ValueError` matching `even`, and `tmp_path` stays empty. Only this catches a size `yuv420p` cannot hold reaching the encoder.
  - `test_encode_refuses_a_frame_of_the_wrong_length`: a second frame one byte short raises `ValueError` matching `frame 2 has 12287 bytes, expected 12288`, and `tmp_path` stays empty. Only this catches a short frame being piped, which would shear every later frame.
  - `test_a_failed_encode_leaves_the_old_clip_and_nothing_else`: `tmp_path / "clip.mp4"` holds `b"old"`; the encoder is an executable script, written by the test, that copies nothing and fails:

    ```sh
    #!/bin/sh
    for last; do :; done
    printf partial > "$last"
    echo "boom" >&2
    exit 3
    ```

    `encode` raises `EncoderError` whose message contains `boom`, `3` and `clip.mp4`; afterwards `sorted(p.name for p in tmp_path.iterdir()) == ["clip.mp4", "fake-encoder"]` and the clip still reads `b"old"`. Only this catches writing to the final name directly, or leaving the temporary file.
  - `test_an_error_while_drawing_leaves_no_clip`: the frames come from a generator that yields one frame and then raises `ValueError("drawing failed")`, with the real encoder; that `ValueError` is raised, and `tmp_path` is empty. Only this catches a half-encoded clip being moved into place when the caller fails.
  - `test_encode_looks_for_the_encoder_first`: `clocks._video.find_encoder` patched to raise `EncoderNotFound("none")`; the frames come from a generator whose first step is `pytest.fail("a frame was asked for")`; `encode` raises `EncoderNotFound`. Only this catches a clip being drawn before the missing encoder is noticed.
  - `test_parse_stream_line` (IDs and inputs): `ffmpeg_9` the line of Q2 → `("h264", "yuv420p", 1500, 900, 20.0)`; `with_colour_and_aspect` `"  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1500x1100 [SAR 1:1 DAR 15:11], 330 kb/s, 20 fps, 20 tbr, 10240 tbn (default)"` → `("h264", "yuv420p", 1500, 1100, 20.0)`; `mpeg4` (measured, Q2) `"  Stream #0:0[0x1](und): Video: mpeg4 (Simple Profile) (mp4v / 0x7634706D), yuv420p, 64x48 [SAR 1:1 DAR 4:3], 5 kb/s, 20 fps, 20 tbr, 10240 tbn (default)"` → `("mpeg4", "yuv420p", 64, 48, 20.0)`; `yuv444p` (measured, Q2) `"  Stream #0:0[0x1](und): Video: h264 (High 4:4:4 Predictive) (avc1 / 0x31637661), yuv444p(progressive), 64x48, 19 kb/s, 20 fps, 20 tbr, 10240 tbn (default)"` → `("h264", "yuv444p", 64, 48, 20.0)`; `fractional_rate` the first with `29.97 fps` (and `20 tbr` left) → 29.97; `two_video_lines` the first followed by the decoder's own report of its null output, `"  Stream #0:0(und): Video: wrapped_avframe, yuv420p(progressive), 1500x900, q=2-31, 200 kb/s, 20 fps, 20 tbn (default)"` → the first line's values. The second, the fractional and the two-line inputs are constructed, not measured (a real decode does print a second `Video:` line: Q2). Only this catches a pattern that takes `High` for the pixel format, the hexadecimal tag for the size, `tbr` for the rate, or the output's line for the input's.
  - `test_parse_stream_line_refuses_other_text` (IDs `audio_only` `"  Stream #0:0: Audio: aac, 44100 Hz"`, `no_rate` the Q2 line with `, 20 fps` removed): `ValueError` whose message contains the offending text. Only this catches a build's rewording being read as a pass.
  - `test_box_order` on bytes built in the test with `struct.pack(">I4s", size, kind) + payload`: `ftyp`, `moov`, `free`, `mdat` → those four; a 64-bit box (`size` 1, then `struct.pack(">Q", 24)`) followed by `moov` → both; a last box whose stated size runs past the end → still listed. Only this catches a walk that stops at, or loops on, the special sizes.
  - `test_probe_counts_the_frames_a_truncated_clip_decodes_to`: sixty frames of 64 by 48 seeded noise (`np.random.default_rng(0).integers(0, 256, (48, 64, 4), np.uint8)` with alpha set to 255, a fresh draw per frame), each written once; the file cut to half its bytes. `probe` returns `codec == "h264"`, `moov_first is True` and `0 < frames < 60`. The exact count depends on the build and is not asserted. Only this catches a frame count read from the container's tables, which a cut file keeps whole.
  - `test_probe_sees_moov_after_mdat`: the small clip copied by the encoder without the flag (`[find_encoder(), "-loglevel", "error", "-y", "-i", clip, "-c", "copy", other]`): `probe(other).moov_first is False` and `frames == 7`. Only this catches `moov_first` hard-wired to true.
  - `test_probe_refuses_a_file_that_is_not_video`: a text file named `clip.mp4` raises `ValueError` matching `no video stream`. Only this catches a probe that returns defaults for an unreadable file.
  - `test_encoder_version_is_the_first_line`: `encoder_version(find_encoder())` starts with `ffmpeg version `. Only this catches a version line taken from standard error or left empty.

  In `tests/test_style.py`:

  - `test_frame_rgba_is_what_save_png_writes` (IDs `plain`, and `hostile` under `matplotlib.rc_context({"savefig.bbox": "tight", "savefig.transparent": True, "savefig.facecolor": "magenta"})`): a figure made inside `with style("light"):` with `plt.subplots(figsize=(7.5, 4.5))` and one line; after that block, `frame_rgba(fig)` has `1500 * 900 * 4` bytes; reshaped to (900, 1500, 4) it equals `np.round(imread(path, format="png") * 255)` for the file `save_png(fig, path)` wrote in the same case; and its first pixel is the light background with alpha 255. Only this catches frames at another resolution, cropped, transparent or on another background than the poster.

  In `tests/test_video.py` also: `test_the_encoder_is_a_development_dependency_only`: with `tomllib`, some entry of `pyproject.toml`'s `dependency-groups.dev` starts with `imageio-ffmpeg`, and no entry of `project.dependencies` starts with `imageio`. Only this catches the encoder becoming a requirement of the installed library.

- [ ] **Step 2: Stage the names, then run.** Create `_video.py` with the classes, `FPS`, `dwell_frames` returning `[24] * (n + 1)`, `find_encoder` returning `""`, `encode` writing nothing and returning 0, `parse_stream_line` returning `("", "", 0, 0, 0.0)`, `box_order` returning `[]`, `probe` returning a `StreamInfo` of those, `encoder_version` returning `""`; add `frame_rgba` returning `b""`. Run `uv run pytest tests/test_video.py tests/test_style.py -q`: every new test fails, none on a missing name (the three that drive the encoder themselves stop on the empty encoder path), except `test_video_module_imports_only_the_standard_library`, which passes already.

- [ ] **Step 3: Implement** as Interfaces says. Run `uv add --dev "imageio-ffmpeg>=0.8.1"` and confirm `git diff --stat` shows `pyproject.toml` and `uv.lock` only for that command.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Add the clip encoder: dwell schedule, raw-frame encode, decode check`. The body gives the encoder's version line here.

---

### Task 3: The clip parts, the `demo_2d` clip, and `render_clip`

**Files:**
- Create: `src/clocks/_clip_parts.py`, `src/clocks/_clip_maps.py`, `src/clocks/_clips.py`, `tests/_clip_replays.py`, `tests/test_clips.py`
- Modify: `src/clocks/viz.py`, `tests/_figure_artists.py`, `tests/test_site_figure_style.py`

**Interfaces:**
- Consumes: `Replay`, `DEMOS` from `clocks._replay`; `Palette`, `style`, `frame_rgba` from `clocks._style`; `trace_limits`, `trace_header` from `clocks._figures`; `dwell_frames`, `find_encoder`, `encode` from `clocks._video`; `LAYOUTS["demo_2d"]` from Task 1's probe script.
- Produces, in `src/clocks/_clip_parts.py`:

```python
Rect = tuple[float, float, float, float]  # left, bottom, width, height

class Clip:
    """A figure with fixed axes, and the function that sets its artists to one row."""
    def __init__(self, fig: Figure, update: Callable[[int, float | None], None]) -> None: ...
    fig: Figure
    def show(self, row: int, azimuth: float | None = None) -> None: ...

def new_figure(height: int) -> Figure: ...
def status_text(row: int) -> str: ...
def draw_status(fig: Figure, colors: Palette) -> Callable[[int], None]: ...
def standard_deviations(replay: Replay) -> NDArray[np.float64]: ...
def draw_traces(fig: Figure, replay: Replay, colors: Palette,
                rects: Mapping[str, Rect]) -> Callable[[int], None]: ...
def draw_dots(ax: Axes, color: str, *, gid: str = "dots", marker: str = "o") -> Line2D: ...
def ellipse_points(mean: NDArray, cov: NDArray) -> NDArray[np.float64]: ...
def draw_ellipse(ax: Axes, color: str, *, gid: str = "ellipse") -> Line2D: ...
def draw_truth(ax: Axes, x: float, y: float, color: str, *, gid: str = "truth") -> Line2D: ...
def zoom_window(truth: NDArray, mean: NDArray, sd: NDArray,
                prior_ranges: Sequence[Sequence[float]]) -> NDArray[np.float64]: ...
def zoom_label(x_factor: float, y_factor: float, y_name: str) -> str: ...
def draw_inset(ax: Axes, rect: Rect, window: NDArray, colors: Palette, *,
               gid: str = "inset") -> Axes: ...
def running_slowing(observations: NDArray) -> NDArray[np.float64]: ...
```

  - `new_figure(height)` returns `plt.figure(figsize=(7.5, height / 100))` after `fig.set_layout_engine("none")` (Q3; `layout="none"` at creation is not enough).
  - `status_text(0) == "Before any observation"`, `status_text(1) == "After 1 observation"`, `status_text(k) == f"After {k} observations"`. `draw_status` adds a figure-level text with gid `"status"`, colour `ink`, anchored top left, and returns the function that sets it for a row.
  - `standard_deviations(replay)` is the square root of `cov`'s diagonal, shape (T + 1, P).
  - `draw_traces` adds one axes per key of `rects`, each key `"trace-<name>"` for a name in `meta["param_names"]`, with that gid. In each: a `Line2D` `"trace-mean"` (`belief`); a band `"trace-band"` (`belief`, face alpha above 0) from mean − sd to mean + sd; a `Line2D` `"trace-truth"` (`truth`, solid) from `[0, T]` at the truth; x-limits (0, T); y-limits `trace_limits(mean column, sd column, truth, prior range)`. The returned function, for a row k, gives the mean line the points 0..k, redraws the band over 0..k (remove it and add it again: Matplotlib 3.9, the oldest the package declares, cannot reset a band's data), and sets the title to `trace_header(name, truth, mean[k], sd[k], prior range)`.
  - `draw_dots` returns a marker-only `Line2D` (`linestyle="none"`, the marker given, a marker size above 0, an alpha strictly between 0 and 1) with no data yet.
  - `ellipse_points(mean, cov)` takes a 2-vector and a 2 by 2 block and returns 121 points, the last equal to the first: `mean + 2 * (V * sqrt(clip(w, 0, None))) @ [cos t, sin t]` for `w, V = np.linalg.eigh(cov)` and t evenly spaced over a full turn. A zero block gives 121 copies of `mean`.
  - `draw_truth` returns a marker-only `Line2D` at (x, y): an open ring (`marker="o"`, `markerfacecolor="none"`, edge in the colour given).
  - `zoom_window(truth, mean, sd, prior_ranges)` takes one entry per axis and returns per axis `(truth - h, truth + h)` with `h = 1.25 * (abs(mean - truth) + 2 * sd)`, or `0.01 * (high - low)` of that axis's prior range where `h` would be 0.
  - `zoom_label(x_factor, y_factor, y_name)`: `f"zoom ×{round(x_factor)}"` when the two round to the same integer, else `f"zoom ×{round(x_factor)} (x), ×{round(y_factor)} ({y_name})"`.
  - `draw_inset` adds a child axes of `ax` at `rect` (in fractions of `ax`) with the given gid and the window as its limits, no tick labels, spines in `muted`.
  - `running_slowing(observations)` returns shape (T + 1, C): row k is the mean of `1 - observations[:k]` over its first axis; row 0 is NaN.
- Produces, in `src/clocks/_clip_maps.py`: `LAYOUT_2D: dict[str, Rect]` with the keys `"map"`, `"inset"`, `"trace-x"`, `"trace-y"`, `"trace-M"`; `clock_fill(running: NDArray, largest: float, colors: Palette) -> list[tuple[float, float, float, float]]`; `build_2d(replay: Replay, colors: Palette) -> Clip`. `clock_fill` returns one RGBA per clock: `background + t * (ink - background)` with `t = min(max(value / largest, 0), 1)` and alpha 1, or `(0, 0, 0, 0)` where the value is NaN.
- Produces, in `src/clocks/_clips.py`:

```python
CLIP_HEIGHTS: dict[str, int] = {
    "demo_1d": 450, "demo_2d": 450, "demo_multi_mass": 500,
    "demo_multi_mass_2d": 550, "demo_model_comparison": 420,
    "demo_echolocation_3d": 550,
}
BUILDERS: dict[str, Callable[[Replay, Palette], Clip]]  # {"demo_2d": build_2d} after this task

class Shot(NamedTuple):
    row: int
    repeat: int
    azimuth: float | None = None

def shots(replay: Replay) -> list[Shot]: ...
def build_clip(replay: Replay, colors: Palette) -> Clip: ...
def final_frame(replay: Replay, colors: Palette) -> Figure: ...
def render_clip(replay: Replay, path: Path, *, theme: str = "light",
                encoder: str | None = None) -> int: ...
```

  - `shots(replay)` is `[Shot(k, n) for k, n in enumerate(dwell_frames(T))]` (Task 6 adds the camera for one demo).
  - `build_clip` raises `ValueError(f"no clip for demo {demo!r}")` for a demo with no builder. `final_frame` builds the clip, shows the last shot and returns the figure; the caller saves and closes it. Both must be called inside `with style(theme) as colors:`.
  - `render_clip` finds the encoder first (when none is passed); then, inside `with style(theme) as colors:`, builds the clip and gives `encode` one `(frame_rgba(clip.fig), shot.repeat)` per shot, each taken after `clip.show(shot.row, shot.azimuth)`, with `width=1500` and `height=2 * CLIP_HEIGHTS[demo]`; closes the figure whether or not the encode succeeded; returns `encode`'s count. It writes what `encode` writes and nothing else.
- `clocks.viz` gains `render_clip` in its imports and `__all__`, and its docstring names `clocks._clips`.
- Produces, in `tests/_figure_artists.py`: `all_axes(fig) -> list[Axes]` (each of `fig.axes` followed by its `child_axes`); `off_palette(fig, colors) -> set[str]` (the `visible_colours` that are neither in `palette_hexes(colors)` nor within 1.5/255 per channel of a point on the segment from `background` to `ink`); `opaque_fills_above(ax, artist) -> list[Artist]` (visible artists drawn directly on `ax`, with a higher `zorder` than `artist`, whose face is filled with alpha 1; the axes' own background patch and its child axes are not counted: an inset covers the part of its panel it sits on by design, and Task 1's sheets check that it covers neither the truth nor the final cloud).
- Produces, in `tests/_clip_replays.py` (a helper module, not collected), each result cached per name: `small(name) -> Replay`, `importlib.import_module(f"clocks._demos.{name}").record(n_observations=9, n_particles=40)`; `shortest(name)`, `record(n_observations=1, n_particles=16)`; `one_particle(name)`, `record(n_observations=2, n_particles=1)`; `committed(name)`, `load_replay` of the committed archive; `edited(replay, **arrays) -> Replay`, a copy with those arrays replaced; and the tables the shared tests read, which Tasks 4 to 6 extend:

```python
# demo -> (axes gid, artist gid, (x column, y column) of cloud, palette role)
DOTS = {"demo_2d": [("map", "dots", (0, 1), "belief")]}
# demo -> (axes gid, artist gid, (i, j): the block of mean and cov)
ELLIPSES = {"demo_2d": [("map", "ellipse", (0, 1))]}
# demo -> (axes gid, artist gid, (i, j): the entries of true_params)
TRUTHS = {"demo_2d": [("map", "truth", (0, 1))]}
# demo -> (parent axes gid, inset gid, (i, j), y name, the parent artists it repeats)
INSETS = {"demo_2d": [("map", "inset", (0, 1), "y", ("dots", "truth", "ellipse"))]}
```

**The `demo_2d` clip's artists.** Axes, in `fig.axes` order: `"map"`, `"trace-x"`, `"trace-y"`, `"trace-M"`; `"inset"` is a child of `"map"`. `"map"` has both limits at the prior range of x and of y and an equal aspect, on a rectangle that is square in inches. In it: `"dots"` (`draw_dots`, `belief`), row k's `cloud[k][:, 0]` against `cloud[k][:, 1]`, cast to float64; `"clocks"`, a scatter collection of squares at `clock_positions` with `ink` edges and faces from `clock_fill(running_slowing(observations)[k], largest, colors)`, `largest` being the largest finite value of the whole `running_slowing` array; `"truth"` (`draw_truth`) at the true (x, y); `"ellipse"` (`draw_ellipse`, `belief`) holding `ellipse_points(mean[k][:2], cov[k][:2, :2])`; one text each reading `covariance, 2 s.d.` (`belief`) and `stronger fill = slower clock` (`ink`). `"inset"` has the limits `zoom_window` gives for (x, y) at the last row and holds `"inset-dots"`, `"inset-truth"` and `"inset-ellipse"` with the same data as their namesakes, and one text, `zoom_label` of the two magnifications: for each axis, the inset's size as a fraction of the map's (the width or height of its rectangle) times the map's span divided by the window's. The cloud's `zorder` is above the clocks' and below the truth's.

- [ ] **Step 1: Write the failing tests** in `tests/test_clips.py`. `CLIPS = sorted(BUILDERS)`; tests marked "(each clip)" are parametrized over it, "(each theme)" over `THEMES`, and "(each cloud)", "(each ellipse)", "(each truth)", "(each inset)" over the entries of the tables for the clips in `CLIPS`. "Built" means `build_clip(replay, colors)` inside `with style(theme) as colors:`, the figure closed in a `finally`; "at row k" means after `clip.show(k)` (with the azimuth of `shots` where the clip has a camera). Unless a test says otherwise the replay is `small(name)` and the theme light. Artists are fetched with `one(axes, gid)` on the axes that owns them, axes with a lookup over `all_axes(fig)` by gid.

  Parts, on literals:

  - `test_status_text`: the three values of Interfaces, and `status_text(50) == "After 50 observations"`. Only this catches the plural at k = 1.
  - `test_running_slowing`: for `np.array([[0.99, 0.98], [0.97, 1.00]])`, row 0 is all NaN and rows 1 and 2 are `[0.01, 0.02]` and `[0.02, 0.01]` (`rtol=1e-12`). Only this catches a mean of rates instead of slowing, a running sum, or a first row of zeros.
  - `test_ellipse_points_lie_two_standard_deviations_out`: `mean = [1.0, -2.0]`, `cov = [[0.04, 0.018], [0.018, 0.09]]`: 121 points, the first equal to the last (`atol=1e-12`), and for every point p, `(p - mean) @ inv(cov) @ (p - mean) == 4` (`rtol=1e-9`); the points' polar angles about `mean`, unwrapped, span a full turn. Only this catches one standard deviation, a diagonal-only covariance (which fails wherever the correlation is not 0), or an arc.
  - `test_ellipse_points_of_no_spread`: a zero block gives 121 copies of the mean, under `warnings.simplefilter("error")`. Only this catches a decomposition that fails on a one-particle run.
  - `test_zoom_window`: truth `[1.5, -1.0]`, mean `[1.42, -1.22]`, sd `[0.11, 0.12]`, prior ranges `[[-8, 8], [-8, 8]]` → `[[1.5 - 0.375, 1.5 + 0.375], [-1.0 - 0.575, -1.0 + 0.575]]` (`rtol=1e-12`); and with mean equal to truth and sd 0 on the first axis only, that axis is `[1.5 - 0.16, 1.5 + 0.16]`. Only this catches a window centred on the mean, one standard deviation instead of two, or a zero-width window.
  - `test_zoom_label`: `(12.2, 11.8, "y")` → `"zoom ×12"`; `(12.4, 8.6, "M")` → `"zoom ×12 (x), ×9 (M)"`. Only this catches one number shown where the two differ.
  - `test_clock_fill` (each theme): for values `[0.0, 0.5, 1.0, 2.0, -0.5, nan]` and `largest=1.0`: the background, the midpoint of background and ink, the ink, the ink, the background, each with alpha 1 (`atol=1e-12`), then `(0, 0, 0, 0)`. Only this catches a blend that runs past either end, which Review Focus 5 describes.

  Every clip:

  - `test_frame_is_1500_by_twice_the_declared_height` (each clip): `tuple(fig.get_size_inches()) == (7.5, CLIP_HEIGHTS[name] / 100)` and `len(frame_rgba(fig)) == 1500 * 2 * CLIP_HEIGHTS[name] * 4`. Only this catches a builder whose canvas disagrees with the declared height.
  - `test_axes_never_move` (each clip): after `fig.canvas.draw()` at rows 0, 1, 4 and 9, the mapping of every axes gid in `all_axes(fig)` to `(ax.get_position().bounds, ax.get_xlim(), ax.get_ylim())` (and `get_zlim()` where there is one) equals the mapping at row 0; the mapping is not empty, no gid is `None` and none repeats. Only this catches autoscaling in a middle frame.
  - `test_panels_are_on_the_canvas_and_apart` (each clip): every axes of `fig.axes` has a rectangle inside the unit square, and no two overlap (the intersection of their rectangles has zero area, to 1e-9). Only this catches a layout table with two panels on top of each other.
  - `test_the_layout_engine_stays_off` (each clip): under `warnings.simplefilter("error")`, `frame_rgba(fig)` at rows 0, 1 and 2; then `type(fig.get_layout_engine()).__name__ == "PlaceHolderLayoutEngine"`. Only this catches a figure made with `layout="none"` (Review Focus 1).
  - `test_status_line` (each clip, each theme): the `"status"` text reads `"Before any observation"`, `"After 1 observation"` and `"After 9 observations"` at rows 0, 1 and 9; it is visible, `ink`, and its position in figure fractions has x below 0.1 and y above 0.9. Only this catches a status line left on the first row.
  - `test_colours_are_the_palette_or_an_ink_blend` (each clip, each theme): at rows 0 and 9, `off_palette(fig, colors) == set()`. Only this catches a default colour: a grey 3D pane, a grey inset edge, an orange clock.
  - `test_dots_are_the_archive_row` (each cloud): at rows 0, 4 and 9, the line's x and y data equal `cloud[row][:, cx]` and `cloud[row][:, cy]` cast to float64, exactly; `assert_marks_drawn(line)`; its colour is the role's; `opaque_fills_above(axes, line) == []`. Only this catches x and y swapped, a stale row, an invisible cloud on frame 0, or an opaque artist drawn over the cloud.
  - `test_traces_follow_the_archive` (each clip that has traces, each theme): for each parameter j, at rows 0, 4 and 9: `"trace-mean"` has x data `arange(row + 1)` and y data `mean[: row + 1, j]`; `band_edges` at those x are `mean ∓ sd` (`rtol=1e-12`); `"trace-truth"` has x data `[0, T]`, y data the truth twice and line style `"-"`; x-limits (0, T); y-limits `trace_limits(...)` (`rel=1e-12`); the title is `trace_header(name, truth, mean[row, j], sd[row, j], prior range)`; colours `belief`, `belief`, `truth`. Only this catches a trace drawn whole from the first frame, a header frozen at the last row, or limits that include the prior row.
  - `test_truth_marks_sit_on_the_truth` (each truth, each theme): the artist's data are `[true_params[i]]`, `[true_params[j]]`; its edge colour is `truth` and its face is `"none"`; it is visible with a marker size above 0. Only this catches a truth mark placed at the final estimate.
  - `test_ellipse_is_the_two_standard_deviation_contour` (each ellipse): at rows 1, 4 and 9 the line holds 121 points p with `(p - m) @ inv(S) @ (p - m) == 4` (`rtol=1e-9`) for `m = mean[row][[i, j]]` and `S = cov[row][[i, j]][:, [i, j]]`; its colour is `belief` (or the cloud's role); its axes holds exactly one visible text reading `covariance, 2 s.d.`. Only this catches an ellipse from another row or another pair of parameters.
  - `test_inset_is_the_fixed_window_on_the_truth` (each inset): the inset's limits equal `zoom_window(...)` computed in the test from the last row (`rtol=1e-12`); at rows 0, 4 and 9 each `"inset-<gid>"` artist has the same data as `"<gid>"` in the parent; the inset holds exactly one text, equal to `zoom_label` of magnifications computed in the test from the two axes' positions and limits. Only this catches a window that follows the cloud, or an inset showing another row than its panel.
  - `test_changing_the_last_mean_moves_the_trace_and_the_header` (each clip that has traces): on `committed(name)` and on a copy with 0.01 times the first parameter's prior width added to `mean[-1, 0]`: at the last row the first trace's last point differs by that amount (`rel=1e-9`), every earlier point and every other trace's data are `np.array_equal`, and the first trace's title differs. Only this catches a renderer that reads another archive, or cached values, for its traces.
  - `test_changing_a_middle_cloud_row_moves_only_that_rows_dots` (each cloud): on `committed(name)` and a copy with 0.5 added to every entry of `cloud[T // 2]`: at row `T // 2` the dots differ; at rows 0, `T // 2 - 1`, `T // 2 + 1` and T they are `np.array_equal`. Only this catches dots taken from a neighbouring row.
  - `test_a_one_observation_frame_is_not_blank` (each clip): for `shortest(name)`, `final_frame` gives a frame with more than 100 distinct RGBA values (a canvas with nothing drawn has 1: Q3). Only this catches a final frame saved before anything is drawn on it.
  - `test_a_one_particle_run_draws` (each clip): for `one_particle(name)`, under `warnings.simplefilter("error")`, the clip builds and `frame_rgba` succeeds at rows 0, 1 and 2. Only this catches a zero-width window, equal axis limits or a failed decomposition (Review Focus 2).
  - `test_shots_follow_the_dwell_schedule` (each clip without a camera): `shots(small(name)) == [Shot(k, n, None) for k, n in enumerate(dwell_frames(9))]`. Only this catches a clip whose frame count is not the schedule's.
  - `test_final_frame_shows_the_last_row` (each clip): the figure `final_frame(small(name), colors)` returns has the status `"After 9 observations"`. Only this catches a poster of the prior.
  - `test_render_clip_encodes_every_shot` (each clip): `render_clip(two, tmp_path / "clip.mp4")` for `two = record(n_observations=2, n_particles=16)` returns 92, and `probe` of the file is `StreamInfo("h264", "yuv420p", 1500, 2 * CLIP_HEIGHTS[name], 20.0, 92, True)`. Only this catches frames of the wrong size reaching the encoder, or shots left out.
  - `test_render_clip_draws_in_the_theme_asked_for` (each theme): with `clocks._clips.encode` replaced by a stand-in that keeps the first frame it is given and returns 0, `render_clip(small("demo_2d"), tmp_path / "x.mp4", theme=theme, encoder="unused")`: the first pixel of that frame is `palette(theme).background`. Only this catches a clip always drawn light.
  - `test_render_clip_closes_its_figure_when_the_encoder_fails`: with `clocks._clips.encode` replaced by a stand-in that raises `EncoderError("no")`, `render_clip` raises it and `plt.get_fignums()` is what it was before. Only this catches a leaked figure per failed clip.
  - `test_render_clip_looks_for_the_encoder_before_drawing`: `clocks._clips.find_encoder` raises `EncoderNotFound("none")` and `clocks._clips.build_clip` is replaced by `pytest.fail`; `render_clip` raises `EncoderNotFound`. Only this catches a clip drawn before the encoder is known to exist.
  - `test_build_clip_refuses_a_demo_without_a_clip`: `build_clip(committed("demo_density"), colors)` raises `ValueError` matching `no clip for demo 'demo_density'`. Only this catches a default builder drawing a wrong clip.

  The `demo_2d` clip:

  - `test_2d_map_spans_the_prior_and_is_square`: both limits (−8.0, 8.0); `ax.get_aspect() == 1.0`; `7.5 * width == CLIP_HEIGHTS["demo_2d"] / 100 * height` for the map's rectangle (`rel=1e-9`). Only this catches a map that Matplotlib must shrink to keep its aspect, which moves the axes on the first draw.
  - `test_2d_clock_fills_follow_the_running_mean` (each theme), on `committed("demo_2d")`, with `running = running_slowing(observations)` and `largest = np.nanmax(running)` computed in the test: at row 0 every face has alpha 0 and every edge is `ink`; at the last row clock c's face is `background + (running[-1, c] / largest) * (ink - background)` (`atol=1e-9`), and the largest of those ratios is below 1 (asserted first: it is 0.848 today, Q8, and 0.766 at M = 0.10, Q6); at the row and clock of `np.nanargmax(running)` the face is `ink`. Only this catches a scale taken from the last row's largest mean, under which the darkest final clock would be `ink`.
  - `test_2d_a_reading_above_one_gives_a_background_fill`: on a copy of `small("demo_2d")` with `observations[0, 2] = 1.02`, at row 1 clock 2's face is the background (`atol=1e-12`), under `warnings.simplefilter("error")`. Only this catches an unclipped blend on a real figure.
  - `test_2d_labels`: the map holds exactly one visible text reading `stronger fill = slower clock`, in `ink`. Only this catches the key being dropped.
  - `test_viz_exports_render_clip`: `clocks.viz.render_clip is clocks._clips.render_clip`, and the name is in `clocks.viz.__all__`.

  In `tests/test_site_figure_style.py`: `LIBRARY_DRAWING_MODULES` gains `"src/clocks/_clip_parts.py"`, `"src/clocks/_clip_maps.py"` and `"src/clocks/_clips.py"`; new `test_every_clip_module_is_scanned`: every `src/clocks/_clip*.py` is in `LIBRARY_DRAWING_MODULES` (only this catches a later clip module escaping the colour check); new `test_clip_modules_import_no_model_code`: for each `src/clocks/_clip*.py`, `_video.py` and (from Task 8) `_site_media.py`, the modules it imports from `clocks` are among `clocks._replay`, `clocks._style`, `clocks._figures`, `clocks._video` and the `clocks._clip*` modules (only this catches a renderer calling the forward model).

- [ ] **Step 2: Stage the names, then run.** Create the three modules with the names of Interfaces: the pure functions raising `NotImplementedError`, `build_2d` returning `Clip(new_figure(450), lambda row, azimuth: None)` with `new_figure` implemented, `shots` returning `[]`, `final_frame` returning the built clip's figure, `render_clip` returning 0; add the helpers and tables to the two test helper modules. Run `uv run pytest tests/test_clips.py tests/test_site_figure_style.py -q`: the literal tests fail on `NotImplementedError`, the artist tests in `one(...)` on missing gids, `test_axes_never_move` on an empty mapping, `test_a_one_observation_frame_is_not_blank` on 1 distinct value, the shot and encode tests on their counts. Four tests that state a property of whatever is drawn pass on the empty figure and first bite in Step 3: the size test, `test_the_layout_engine_stays_off`, `test_panels_are_on_the_canvas_and_apart` and `test_colours_are_the_palette_or_an_ink_blend`; so do `test_a_one_particle_run_draws` and the two guard tests.

- [ ] **Step 3: Implement** as Interfaces and "The `demo_2d` clip's artists" say, with `LAYOUT_2D` copied from `LAYOUTS["demo_2d"]`. Keyword styling only: these modules are read by the source guard.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Draw the 2D clip from its replay: fixed panels, traces, ellipse, inset; render_clip`.

---

### Task 4: The two track clips: `demo_1d` and `demo_multi_mass`

**Files:**
- Create: `src/clocks/_clip_tracks.py`
- Modify: `src/clocks/_clips.py`, `tests/_clip_replays.py`, `tests/test_site_figure_style.py`
- Test: `tests/test_clips.py`

**Interfaces:**
- Consumes: everything Task 3 produced in `_clip_parts.py`; `LAYOUTS["demo_1d"]` and `LAYOUTS["demo_multi_mass"]` from Task 1.
- Produces, in `src/clocks/_clip_tracks.py`: `LAYOUT_1D` and `LAYOUT_MULTI_1D` (`dict[str, Rect]`); `SIDE_OFFSET: float`, the distance in x between a clock's three marks, from Task 1; `slowing_limits(replay: Replay) -> tuple[float, float]`; `build_1d(replay, colors) -> Clip`; `build_multi_1d(replay, colors) -> Clip`. `BUILDERS` gains both. `slowing_limits` returns the smallest interval holding `1 - true_rates`, `1 - predicted_mean ± predicted_std` for every row, and `running_slowing(observations) ± noise_std / sqrt(k)` for every row k ≥ 1, widened by 4% of its width on each side.
- `tests/_clip_replays.py`: `DOTS` gains `"demo_1d": [("cloud", "dots", (0, 1), "belief")]` and `"demo_multi_mass": [("cloud", "dots-1", (0, 2), "belief"), ("cloud", "dots-2", (1, 3), "belief_alt")]`; `ELLIPSES` gains `("cloud", "ellipse", (0, 1))` and `("cloud", "ellipse-1", (0, 2))`, `("cloud", "ellipse-2", (1, 3))`; `TRUTHS` the same pairs under `"truth"`, `"truth-1"`, `"truth-2"`; `INSETS` gains `("cloud", "inset", (0, 1), "M", ("dots", "truth", "ellipse"))` and, for the two masses, `("cloud", "inset-1", (0, 2), "M", ("dots-1", "truth-1", "ellipse-1"))` and `("cloud", "inset-2", (1, 3), "M", ("dots-2", "truth-2", "ellipse-2"))`. An inset's copies carry the gid `"<inset gid>-<artist gid>"` (so `"inset-dots"` in the one-mass clips, as in Task 3, and `"inset-1-dots-1"` here); Task 3's inset test reads the names from the table.

**The track clips' artists.** Axes, in `fig.axes` order: `"cloud"`, `"strip"`, `"slowing"`, then the traces (`"trace-x"`, `"trace-M"`; or `"trace-x₁"`, `"trace-x₂"`, `"trace-M₁"`, `"trace-M₂"`, stacked). `"cloud"` has the prior range of x as its x-limits and the mass prior range as its y-limits; `"strip"` and `"slowing"` have the same x-limits and the same left edge and width as `"cloud"`. In `"cloud"`: the dots, truths, ellipses and insets of the tables above, the second mass drawn with `marker="^"` in `belief_alt`; for two masses, one text `mass 1` and one `mass 2`, each beside its truth, in `truth`. In `"strip"`: `"clocks"`, a marker-only `Line2D` of `ink` squares at the clocks' x and y = 0. In `"slowing"`, with x the clocks' positions and d = `SIDE_OFFSET`: `"slowing-truth"`, a marker-only `Line2D` (`marker="_"`, `truth`) at (x − d, `1 - true_rates`); `"slowing-predicted"`, a marker-only `Line2D` (`belief`) at (x, `1 - predicted_mean[k]`), and `"slowing-predicted-bar"`, the `vlines` collection (`belief`) from that value minus to plus `predicted_std[k]`; `"slowing-observed"`, a marker-only `Line2D` (`ink`) at (x + d, `running_slowing(observations)[k]`), and `"slowing-observed-bar"`, the `vlines` collection (`ink`) of half-length `noise_std / sqrt(k)`; the two observed artists are not visible at row 0. y-limits `slowing_limits(replay)`; y label `slowing (1 − rate)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_clips.py`; `TRACKS = ("demo_1d", "demo_multi_mass")`. The shared tests of Task 3 now run for both clips through `CLIPS` and the tables.

  - `test_track_panels_share_the_x_scale` (each of `TRACKS`): `"cloud"`, `"strip"` and `"slowing"` have equal x-limits, the prior's, and rectangles with equal `x0` and `width` (`rel=1e-12`). Only this catches a clock strip that is not under the positions it marks.
  - `test_strip_shows_the_clocks` (each of `TRACKS`, each theme): `"clocks"` has the clocks' x as its x data, marker `"s"`, colour `ink`, and is visible with a marker size above 0. Only this catches clocks drawn from a constant.
  - `test_slowing_marks_follow_the_archive` (each of `TRACKS`, each theme), at rows 0, 1, 4 and 9, with every expected value computed in the test from the replay: `"slowing-truth"` y data `1 - true_rates` (`rtol=1e-12`) in `truth`; `"slowing-predicted"` y data `1 - predicted_mean[row]`, and the segments of `"slowing-predicted-bar"` run from that minus to that plus `predicted_std[row]` at the same x, both `belief`; at row 0 `"slowing-observed"` and `"slowing-observed-bar"` are not visible; at later rows they are, the marks at `running_slowing(observations)[row]` and the segments of half-length `noise_std / sqrt(row)`, both `ink`. Only this catches rates plotted for slowing, bars of one row beside marks of another, a bar that does not shrink with √k, or readings shown before any exist.
  - `test_slowing_marks_stand_side_by_side` (each of `TRACKS`): for every clock, the x of its truth mark, predicted mark and observed mark are `x - d`, `x`, `x + d` for one `d > 0` (`rel=1e-12`), and `2 * d` is less than the smallest gap between clocks. Only this catches the three marks drawn on top of each other.
  - `test_slowing_limits_hold_every_row`: `slowing_limits(small(name))` contains, with room on both sides, every value listed in Interfaces, computed in the test; and with `predicted_std[0]` doubled in a copy the interval widens. Only this catches limits that leave the prior's bars out of view (decision D19).
  - `test_two_masses_are_told_apart_without_colour` (each theme), for `demo_multi_mass`: `"dots-1"` and `"dots-2"` have different markers and the colours `belief` and `belief_alt`; the cloud holds exactly one visible text `mass 1` and one `mass 2`, each in `truth` and each nearer its own truth mark than the other's. Only this catches two clouds that differ by hue alone, or labels swapped.

- [ ] **Step 2: Stage the names, then run.** Add the module with `build_1d` and `build_multi_1d` returning empty clips of their heights, `slowing_limits` returning `(0.0, 1.0)`, and register both; extend the tables. Run `uv run pytest tests/test_clips.py -q`: the new and the shared artist tests fail in `one(...)` or on their values for the two clips.

- [ ] **Step 3: Implement** as "The track clips' artists" says, with the layouts and `SIDE_OFFSET` from Task 1. `tests/test_site_figure_style.py`: `LIBRARY_DRAWING_MODULES` gains `"src/clocks/_clip_tracks.py"`.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Draw the two track clips: cloud, clock strip, per-clock slowing, traces`.

---

### Task 5: The two-mass map and the model-comparison clip

**Files:**
- Create: `src/clocks/_clip_models.py`
- Modify: `src/clocks/_clip_maps.py`, `src/clocks/_clips.py`, `tests/_clip_replays.py`, `tests/test_site_figure_style.py`
- Test: `tests/test_clips.py`

**Interfaces:**
- Consumes: Task 3's parts; `LAYOUTS["demo_multi_mass_2d"]` and `LAYOUTS["demo_model_comparison"]` from Task 1.
- Produces: in `_clip_maps.py`, `LAYOUT_MULTI_2D` and `build_multi_2d(replay, colors) -> Clip`; in `_clip_models.py`, `LAYOUT_MODELS`, `GUTTER: float` (the width, in observations, of the label gutter right of T, from Task 1) and `build_models(replay, colors) -> Clip`. `BUILDERS` gains both.
- `tests/_clip_replays.py`: `DOTS` gains `"demo_multi_mass_2d": [("map", "dots-1", (0, 1), "belief"), ("map", "dots-2", (2, 3), "belief_alt")]`; `ELLIPSES` and `TRUTHS` the blocks (0, 1) and (2, 3) under `-1` and `-2`; `INSETS` gains nothing. A new helper `two_candidates() -> Replay`: `small("demo_model_comparison")` with candidate 1 removed (column 0 of `model_prior`, `model_posterior`, `model_log_evidence` and `model_ess` dropped, `model_ks` `[2, 3]`, `model_prior` and each `model_posterior` row divided by its sum, `final_mean_k1` and `final_std_k1` deleted), which the helper passes through `validate_replay`.

**The two-mass map's artists.** Axes: `"map"` (as in Task 3: the prior ranges, equal aspect, square), then six traces in three rows and two columns, `"trace-x₁"`, `"trace-y₁"`, `"trace-M₁"` in the left column and `"trace-x₂"`, `"trace-y₂"`, `"trace-M₂"` in the right. In the map: the dots, truths and ellipses of the tables, the second mass with `marker="^"` in `belief_alt`; texts `mass 1` and `mass 2` in `truth`, beside their truths; `"clocks"`, a marker-only `Line2D` of `ink` squares at `clock_positions`; one text `covariance, 2 s.d.`. No inset.

**The model-comparison clip's artists.** One axes, `"posterior"`: x-limits (0, T + `GUTTER`), y-limits containing [0, 1], x label `observations`, y label `posterior probability`. For the i-th entry K of `model_ks`: a `Line2D` `f"model-{K}"` in `ordered[i]`, at row k the points 0..k of `model_posterior[:, i]`; a text `f"label-{K}"` in `ordered[i]` reading `f"K = {K}"`, or `f"K = {K} (true model)"` when K is `meta["true_k"]`, at a fixed place in the gutter (x at or beyond T; the labels in `model_ks` order from the top, at least 0.08 apart in y); and a `Line2D` `f"connector-{K}"` in `ordered[i]`, no wider than 1 point, from the line's head (k, `model_posterior[k, i]`) to the label's place. `ValueError` matching `at most three candidates` for more than three.

- [ ] **Step 1: Write the failing tests** in `tests/test_clips.py`. The shared tests of Task 3 run for both clips (the model clip has no cloud, trace, truth, ellipse or inset entries).

  - `test_two_mass_map_has_one_trace_column_per_mass`: the three mass-1 traces share one `x0`, the three mass-2 traces share a larger one, and each row's two traces share a `y0` (`rel=1e-12`). Only this catches the six traces laid out in parameter order across both columns.
  - `test_two_mass_map_has_no_inset_and_plain_clocks` (each theme): `map.child_axes == []`; `"clocks"` has `clock_positions`' columns as data, marker `"s"`, colour `ink`. Only this catches clocks drawn from the first demo's eight positions.
  - `test_two_mass_map_labels_each_truth` (each theme): as `test_two_masses_are_told_apart_without_colour`, on the map.
  - `test_model_lines_grow_from_the_prior` (each theme), at rows 0, 1, 4 and 9, for each candidate: the line's x data are `arange(row + 1)` and its y data `model_posterior[: row + 1, i]`, exactly; at row 0 the single point is `model_prior[i]`; its colour is `colors.ordered[i]`. Only this catches a column read by K instead of by position, or lines that start at the first observation.
  - `test_model_labels_keep_their_places_and_name_the_truth` (each theme): the label texts are `["K = 1", "K = 2 (true model)", "K = 3"]`; their positions are the same at rows 0 and 9, have x ≥ 9 (T of the small replay), and y values strictly decreasing in `model_ks` order and at least 0.08 apart; each label's colour is its line's. With `meta["true_k"]` changed to 3 in a copy (not validated: the builder is called directly), the texts are `["K = 1", "K = 2", "K = 3 (true model)"]`. Only this catches labels that follow the lines' heads and stack where the lines cross, or a hard-coded true model.
  - `test_model_connectors_join_head_to_label`, at rows 0, 4 and 9: each connector's first point is (row, `model_posterior[row, i]`) and its last the label's position; its line width is at most 1 and its colour its line's. Only this catches connectors left pointing at an earlier head.
  - `test_model_colours_go_by_position_in_model_ks` (each theme): for `two_candidates()`, the lines `"model-2"` and `"model-3"` are `ordered[0]` and `ordered[1]`, and the clip builds and draws at every row. Only this catches `ordered[K - 1]`, which agrees with position for the committed candidates.
  - `test_model_clip_refuses_a_fourth_candidate`: a copy with `model_ks` `[1, 2, 3, 4]` and matching columns (not validated: the builder is called directly) raises `ValueError` matching `at most three candidates`. Only this catches a fourth line drawn in a recycled colour.
  - `test_changing_a_posterior_row_moves_only_that_point`, on `committed("demo_model_comparison")` and a copy with the values of `model_posterior[40, 1]` and `[40, 2]` exchanged: at the last row, `"model-2"` and `"model-3"` differ at index 40 only. Only this catches a model clip reading another archive.
  - `test_model_axes`: x-limits `(0, T + GUTTER)`, y-limits containing 0 and 1, the two labels. Only this catches a y-axis that rescales to the lines.

- [ ] **Step 2: Stage the names, then run.** Add `build_multi_2d` and `build_models` returning empty clips of their heights and register them; extend the tables and add `two_candidates`. Run `uv run pytest tests/test_clips.py -q`: the new tests and the shared artist tests fail for the two clips.

- [ ] **Step 3: Implement** as the two artist paragraphs say. `tests/test_site_figure_style.py`: `LIBRARY_DRAWING_MODULES` gains `"src/clocks/_clip_models.py"`.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Draw the two-mass map and the model-comparison clip`.

---

### Task 6: The echolocation clip and its camera

**Files:**
- Create: `src/clocks/_clip_echo.py`
- Modify: `src/clocks/_clips.py`, `tests/_clip_replays.py`, `tests/test_site_figure_style.py`
- Test: `tests/test_clips.py`

**Interfaces:**
- Consumes: Task 3's parts; `LAYOUTS["demo_echolocation_3d"]` from Task 1.
- Produces, in `src/clocks/_clip_echo.py`: `LAYOUT_ECHO`; `ELEVATION = 18.0`; `azimuth(frame: int, frames: int) -> float`, the design's `-75 + 40 * (1 - cos(pi * frame / (frames - 1))) / 2`; `in_view(points: NDArray, bounds: Sequence[float]) -> NDArray[np.bool_]` (true where x, y and z all lie in the closed interval, as `weight_in_view` counts them); `view_weight_text(row: int, weight: float) -> str`; `build_echo(replay, colors) -> Clip`. `BUILDERS` gains it.
- `shots(replay)` for `demo_echolocation_3d` (in `_clips.py`): with `dwell = dwell_frames(T)` and `F = sum(dwell[:-1])`, one `Shot(row, 1, azimuth(f, F))` for each video frame f of rows 0 to T − 1, in order, then `Shot(T, dwell[-1], azimuth(F - 1, F))`. For every other demo, unchanged.
- `tests/_clip_replays.py`: `DOTS` gains `"demo_echolocation_3d": [("overview", "overview-dots", (0, 2), "belief")]`.

**The echolocation clip's artists.** Axes, in `fig.axes` order: `"scene"` (3D), `"overview"`, `"fit"`, `"trace-x"`, `"trace-y"`, `"trace-z"`, `"trace-M"`. Figure-level texts: `"status"` and, under it, `"view-weight"`, reading `view_weight_text(k, weight_in_view[k])`: `f"{100 * w:.1f}% of the prior lies inside the close view"` at row 0 and `f"{100 * w:.1f}% of the cloud's weight lies inside the close view"` afterwards.
  - `"scene"`: all three limits are `view_bounds`; the camera is `view_init(elev=18, azim=<the shot's azimuth>)`, and `azimuth(F - 1, F)` when none is given. `"head"`, the twelve edges of the clocks' bounding box, a line collection in `muted`; `"clocks"`, a marker-only `Line3D` of `ink` squares at `clock_positions`; `"dots"`, a marker-only `Line3D` in `belief` holding exactly the points of `cloud[k]` for which `in_view` is true, in archive order; `"truth"`, an open ring (`marker="o"`, face `"none"`, edge `truth`) at the true (x, y, z), larger than the dots; `"truth-drop"`, a `Line3D` in `truth` from the truth straight down to z = `view_bounds[0]`. Panes, pane edges, grid and axis lines take palette colours.
  - `"overview"`: both limits the prior range of x and of z, equal aspect; `"overview-dots"`, `cloud[k][:, 0]` against `cloud[k][:, 2]`; `"view-box"`, a closed `Line2D` in `muted` around the square from `view_bounds[0]` to `view_bounds[1]`.
  - `"fit"`: equal limits on both axes, fixed, holding for every row k ≥ 1 the running mean of the centred readings, `observations[:k].mean(axis=0)`, and `predicted_mean[k]`; x label `observed mean`, y label `fitted`. `"fit-points"`, a marker-only `Line2D` in `ink`, one point per clock, those two quantities; `"fit-diagonal"`, a `Line2D` in `muted` between the two corners of the limits; `"fit-empty"`, a text `no readings yet`. At row 0 the text is visible and the points are not; afterwards the reverse.

- [ ] **Step 1: Write the failing tests** in `tests/test_clips.py`. The shared tests run for the clip; where they show a row they pass the azimuth of the matching shot.

  - `test_azimuth` on literals (Q5): `azimuth(f, 306)` for f = 0, 76, 152, 229, 305 is −75.0, −69.17850576840875, −55.103002582480016, −40.82149423159126, −35.0 (`abs=1e-9`); `azimuth(f, 24)` for f = 0, 6, 12, 23 is −75.0, −68.65106286437309, −53.63515173270658, −35.0. Only this catches a turn at constant speed, which gives −65.03 at frame 76.
  - `test_echo_shots_turn_the_camera_and_hold`, on `committed("demo_echolocation_3d")`: 307 shots; the first 306 have `repeat == 1`, rows `np.repeat(np.arange(80), dwell_frames(80)[:-1])` and azimuths `azimuth(f, 306)`; the last is `Shot(80, 50, -35.0)` (the azimuth `abs=1e-9`); the repeats sum to 356. And for `shortest(...)`: 25 shots, the last `Shot(1, 50, -35.0)`. Only this catches a camera that keeps turning through the hold, or a turn spread over all 356 frames.
  - `test_echo_camera_follows_the_shot`: after `clip.show(shot.row, shot.azimuth)` for shots 0, 76 and 306 of the committed archive, `scene.azim` is the shot's azimuth (`abs=1e-9`) and `scene.elev == 18`; after `clip.show(80)` with no azimuth, `scene.azim` is −35 (`abs=1e-9`). Only this catches a clip that ignores the azimuth it is given.
  - `test_echo_scene_is_the_fixed_cube`: the three limits equal `view_bounds`. Only this catches a cube recomputed from the row.
  - `test_echo_close_view_holds_exactly_the_dots_inside_the_cube`: on the committed archive, at row 0 `one(scene, "dots").get_data_3d()` holds exactly the 2 points of `cloud[0]` for which all three coordinates lie in `view_bounds` (the test first asserts there are 2 of 500), and at row 1 all 500; on a copy of `small(...)` with the x of the first three dots of row 4 set to `view_bounds[1] + 1.0`, row 4 shows the other 37 in order. Only this catches uncropped points in the close view (Review Focus 5).
  - `test_in_view_counts_the_faces`: for bounds (−1.0, 2.0), the points (−1, 0, 2), (2, 2, 2), (2.0000001, 0, 0) and (0, −1.0000001, 0) give True, True, False, False. Only this catches an open interval, which would disagree with the recorded weight for a dot on a face.
  - `test_echo_overview_spans_the_prior_and_marks_the_view`: limits (−16.0, 16.0) on both axes; `"view-box"`'s x and y data visit the four corners of the square from −1.3 to 4.8 and return to the first, on the committed archive. Only this catches an overview zoomed to the cloud, where the collapse cannot be seen.
  - `test_view_weight_text`: on the committed archive the text is `0.4% of the prior lies inside the close view` at row 0 and `100.0% of the cloud's weight lies inside the close view` at rows 1 and 80; `view_weight_text(3, 0.5)` is `50.0% of the cloud's weight lies inside the close view`. Only this catches the prior's wording used after the first observation, or a percentage computed from the displayed dots.
  - `test_echo_fit_panel` (each theme), with expected values computed in the test: at row 0 `"fit-empty"` is visible and reads `no readings yet`, and `"fit-points"` is not visible; at rows 1, 4 and 9 the text is hidden and the points' x data are `observations[:row].mean(axis=0)` and y data `predicted_mean[row]` (`rtol=1e-12`, `atol=1e-15`), in `ink`; the x- and y-limits are equal; `"fit-diagonal"` runs corner to corner in `muted`; the labels read `observed mean` and `fitted`. Only this catches fitted values on the observed axis, a single reading plotted for the running mean, or unequal scales.
  - `test_echo_truth_ring_and_drop_line` (each theme): `"truth"` is at the true (x, y, z), with face `"none"`, edge `truth` and a marker size above that of `"dots"`; `"truth-drop"` runs from the truth to (x, y, `view_bounds[0]`). Only this catches a filled marker hiding the final cloud.
  - `test_echo_head_and_clocks` (each theme): `"clocks"` holds `clock_positions`' three columns in `ink`; the vertices of `"head"`'s segments are exactly the eight corners of the clocks' bounding box, twelve segments, in `muted`. Only this catches a head drawn at a constant size.
  - `test_every_clip_demo_has_a_builder`: `set(BUILDERS) == set(CLIP_HEIGHTS) == set(DEMOS) - {"demo_density"}`. Only this catches a demo left without a clip.

- [ ] **Step 2: Stage the names, then run.** Add the module with `azimuth` returning −75.0, `in_view` returning all true, `view_weight_text` returning `""` and `build_echo` returning an empty clip of height 550; register it; leave `shots` unchanged. Run `uv run pytest tests/test_clips.py -q`: the new tests fail on their values, and the shared tests for the clip in `one(...)`.

- [ ] **Step 3: Implement** as "The echolocation clip's artists" and Interfaces say. `tests/test_site_figure_style.py`: `LIBRARY_DRAWING_MODULES` gains `"src/clocks/_clip_echo.py"`.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Draw the echolocation clip: close view, whole prior box, fit panel, eased camera`.

---

### Task 7: The rendered-site check learns clips

No page changes: the manifest still lists plots only, so the real site passes as before. The clip branch is exercised by fixtures here and by the real site from Task 10.

**Files:**
- Modify: `scripts/check_site_figures.py`
- Test: `tests/test_check_site_figures.py`

**Interfaces:**
- Consumes: `probe`, `dwell_frames`, `find_encoder`, `EncoderNotFound` from `clocks._video`; `load_replay`, `save_replay` from `clocks._replay`.
- Produces, in `scripts/check_site_figures.py`:
  - `ARCHIVES = ROOT / "assets" / "replays"` and `RESPONSIVE_RULE = "figure.replay video { max-width: 100%; height: auto; }"`.
  - `_FigureParser` also collects: `themed`, every themed element in document order, each a plot wrapper (`_Wrapper`, as today) or a clip rendering (`_Video`: its themes, its attributes as a dict in which a bare attribute such as `controls` is present with the value `None`, and the index of its replay figure); `replay_captions`, one list of caption texts per `<figure>` whose class list holds `replay`; and `texts`, a dict from each `id` on the page to the list of whitespace-normalised text contents of the elements carrying it. A themed `<video>` that opens inside a replay figure and outside any plot wrapper is a clip rendering; **any other themed element is a plot wrapper**, as section 10 says. A `figcaption` inside a replay figure belongs to that figure.
  - `_resolve_source(src, page_path, output_dir, what="image")`: the same rules, with `what` as the first word of each message (`"video"`, `"poster"`).
  - `figure_failures(output_dir: Path) -> list[str]`, signature unchanged. `EXPECTED_FIGURES` is unchanged in this task.

**What the check does with a clip.** In `_page_failures`, after the both-classes, nesting and count checks (which now count every themed element; the count message keeps its words), figure `i`'s two themed elements are checked in this order, and the first failure of the first two ends the checks on that figure:
  1. Kind: `f"{page}: figure {i}: expected a clip, found a plot wrapper"` or `f"{page}: figure {i}: expected a plot, found a clip video"`.
  2. Order: today's `expected light-content then dark-content, found ...`.
  3. A `Fig` whose kind is neither `"plot"` nor `"clip"` raises `ValueError(f"{page}: figure {i}: no check for kind {kind!r}")`, as today.
  4. For a clip, with `label = f"{page}: figure {i}"`, `h = figure.height` and `clip = figure.clip`:
     - `f"{label}: the two videos are not in one <figure class=\"replay\">"`, and nothing more for this figure;
     - the figure's captions: `f"{label}: expected 1 figcaption, found {n}"`, or `f"{label}: empty caption"`;
     - per theme, with `t = f"{label} {theme}-content"`: `f"{t}: width {value}, expected 750"`; `f"{t}: height {value}, expected {h}"`; `f"{t}: no controls attribute"`; `f"{t}: no playsinline attribute"`; `f"{t}: preload {value!r}, expected 'none'"`; `f"{t}: autoplay is set"`; `f"{t}: loop is set"`; `f"{t}: empty aria-label"`; `f"{t}: no aria-describedby"`, or `f"{t}: aria-describedby {name!r} names no element on the page"`, or `f"{t}: aria-describedby {name!r} names {n} elements"`, or `f"{t}: the description {name!r} is empty"`;
     - per theme, the files: `_resolve_source` on `src` (`what="video"`) and on `poster` (`what="poster"`); then `f"{t}: video is {name!r}, expected {clip}.{theme}.mp4"` and `f"{t}: poster is {name!r}, expected {clip}.{theme}.png"` on the resolved files' names; `f"{t}: poster is not a PNG file"`, or `f"{t}: poster is {w} by {h2} pixels, expected 1500 by {2 * h}"` from the PNG header;
     - per theme, the decoded stream, when the video file resolved and an encoder was found: `f"{t}: video cannot be decoded: {error}"` when `probe` raises `ValueError`; else each of `f"{t}: video codec {codec}, expected h264"`, `f"{t}: video pixel format {pix_fmt}, expected yuv420p"`, `f"{t}: video is {w} by {h2} pixels, expected 1500 by {2 * h}"`, `f"{t}: video runs at {fps:g} frames per second, expected 20"`, `f"{t}: video decodes to {frames} frames, expected {n}"`, `f"{t}: video has mdat before moov"` that applies. `n` is `sum(dwell_frames(T))` for the `n_observations` of `ARCHIVES / f"{clip}.npz"`; an archive that cannot be read gives `f"{label}: no readable archive for clip {clip!r}: {error}"` and no frame-count check;
     - the pair: `f"{label}: aria-describedby differs between themes: {a!r} vs {b!r}"`; `f"{label}: light and dark use the same video file"`, else `f"{label}: light and dark videos have the same content"`; the same two for posters, with `poster` for `video`.
  Two checks are made once per site, and only when the manifest lists at least one clip: the encoder is looked up before any page is read, and `EncoderNotFound` gives the single failure `f"clips cannot be decoded: {error}"` (the other clip checks still run); and `output_dir / "styles.css"`, with all whitespace removed, must contain `RESPONSIVE_RULE` with all whitespace removed, else `f"styles.css: the rule {RESPONSIVE_RULE!r} is missing"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_check_site_figures.py`.

  Fixtures. `CLIP = Fig("clip", 450, "demo_2d")`. A module-scoped fixture `clip_media` makes one directory holding: `archives/demo_2d.npz`, `save_replay` of `demo_2d.record(n_observations=2, n_particles=16)`, so 92 frames are expected; `demo_2d.light.mp4` and `demo_2d.dark.mp4`, each 92 different frames of 1500 by 900 piped to `find_encoder()` by a helper in the test file that spells out Task 2's command itself (so a fault in `clocks._video.encode` cannot hide in these fixtures): a solid frame in that theme's background colour with a 64 by 64 block of seeded noise that moves ten pixels a frame (measured: 0.9 s to encode, 249 kB, and cut in half it decodes to 45 frames with exit status 0); and `demo_2d.light.png`, `demo_2d.dark.png`, written with `matplotlib.image.imsave` from a (900, 1500, 3) array of the same colour. A builder `_clip_page(tmp_path, clip_media, *, light=None, dark=None, caption="A caption.", description="Words.", page=PAGE, before="", after="") -> Path` copies the four media files to `tmp_path / "media"`, writes `tmp_path / "styles.css"` holding `RESPONSIVE_RULE`, and writes the page with the markup of the design's section 4: `light` and `dark` are dicts of attribute overrides applied to the default attributes of each `<video>` (a value of `None` removes the attribute; the key `"flags"` replaces the bare attributes `controls playsinline`). An autouse fixture points `check_site_figures.ARCHIVES` at `clip_media / "archives"`; clip tests set the manifest to `{PAGE: (CLIP,)}`. `_single_failure` is used as today.

  - `test_a_well_formed_clip_passes`: `[]`. Only this catches a clip branch that rejects the design's own markup.
  - `test_a_clip_and_two_plots_in_order_pass`: manifest `(Fig("plot", 540), CLIP, Fig("plot", 420))`, a page holding `_pair(1, 540)`, the clip figure, `_pair(2, 420)`: `[]`. Only this catches clip and plot elements being counted in separate sequences.
  - Negative fixtures, each asserting exactly one failure containing the text given, unless stated:
    - `test_a_missing_dark_video` (the dark `<video>` left out): `expected 1 figure(s), 2 theme wrappers; found 1`.
    - `test_a_plot_where_a_clip_is_listed` (the page holds `_pair(1, 450)`): `figure 1: expected a clip, found a plot wrapper`.
    - `test_a_clip_where_a_plot_is_listed` (manifest `(Fig("plot", 420),)`, the page a clip): `figure 1: expected a plot, found a clip video`.
    - `test_a_themed_video_outside_a_replay_figure` (the two videos in a plain `<div>`): `figure 1: expected a clip, found a plot wrapper`.
    - `test_dark_video_before_light`: `figure 1: expected light-content then dark-content, found dark-content then light-content`.
    - `test_videos_in_two_replay_figures`: `the two videos are not in one <figure class="replay">`.
    - `test_a_clip_out_of_order` (manifest `(CLIP, Fig("plot", 420))`, the page a plot pair then the clip): two failures, `figure 1: expected a clip, found a plot wrapper` and `figure 2: expected a plot, found a clip video`.
    - `test_a_clip_on_an_unlisted_page` (a second page, not in the manifest, holding the clip): `story/other.html: not in the figure manifest, but has 2 theme wrapper(s)`.
    - `test_wrong_clip_height` (both videos `height="500"`): two failures, `figure 1 light-content: height 500, expected 450` and the same for dark.
    - `test_wrong_clip_width` (dark `width="700"`): `figure 1 dark-content: width 700, expected 750`.
    - `test_autoplay` (light flags `controls playsinline autoplay`): `figure 1 light-content: autoplay is set`.
    - `test_loop`: `loop is set`. `test_no_controls` (flags `playsinline`): `no controls attribute`. `test_no_playsinline`: `no playsinline attribute`. `test_preload_auto` (dark `preload="auto"`): `preload 'auto', expected 'none'`.
    - `test_missing_aria_label` (light `aria-label` removed) and `test_blank_aria_label` (`aria-label="  "`): `figure 1 light-content: empty aria-label`.
    - `test_missing_description` (the described element left out): two failures, each `aria-describedby 'demo-2d-text' names no element on the page`.
    - `test_empty_description` (`description="  "`): two failures, each `the description 'demo-2d-text' is empty`.
    - `test_description_differs_between_themes` (dark `aria-describedby="other-text"`, with a second non-empty element of that id on the page): `aria-describedby differs between themes: 'demo-2d-text' vs 'other-text'`.
    - `test_two_elements_share_the_description_id`: two failures, each `names 2 elements`.
    - `test_no_caption` (`caption=None`): `expected 1 figcaption, found 0`. `test_empty_clip_caption` (`caption=" "`): `figure 1: empty caption`.
    - `test_missing_poster` (the dark poster file deleted): `figure 1 dark-content: poster '../media/demo_2d.dark.png' does not exist`.
    - `test_wrong_clip_name` (both `src` and `poster` of both themes naming `demo_1d.<theme>.*`, those files present as copies): four failures, `video is 'demo_1d.light.mp4', expected demo_2d.light.mp4`, `poster is 'demo_1d.light.png', expected demo_2d.light.png`, and the same two for dark.
    - `test_wrong_theme_file` (light `src` naming the dark video): `video is 'demo_2d.dark.mp4', expected demo_2d.light.mp4` and `light and dark use the same video file`.
    - `test_identical_light_and_dark_videos` (the dark video's bytes replaced by the light one's): `figure 1: light and dark videos have the same content`. `test_identical_light_and_dark_posters`: `light and dark posters have the same content`.
    - `test_poster_of_the_wrong_size` (the light poster rewritten at (450, 750, 3)): `poster is 750 by 450 pixels, expected 1500 by 900`. `test_poster_that_is_not_a_png` (its bytes `b"JPEG"`): `poster is not a PNG file`.
    - `test_a_truncated_video` (the dark video cut to half its bytes): exactly one failure, matching `figure 1 dark-content: video decodes to (\d+) frames, expected 92`, with the captured number between 1 and 91. The number is not pinned: it depends on the encoder build.
    - `test_a_video_in_another_codec` (the light video re-encoded by the encoder with `-c:v mpeg4 -movflags +faststart`): `video codec mpeg4, expected h264`. `test_a_video_in_another_pixel_format` (`-c:v libx264 -pix_fmt yuv444p -movflags +faststart`): `video pixel format yuv444p, expected yuv420p`. `test_a_video_without_faststart` (copied with `-c copy` and no flag): `video has mdat before moov`. Both builds here report those two re-encodings as the lines quoted in Task 2 (Q2).
    - `test_a_video_of_the_wrong_frame_count` (the manifest's archive replaced by a one-observation recording, so 74 frames are expected): two failures, each `video decodes to 92 frames, expected 74`.
    - `test_a_file_that_is_not_a_video` (the light video's bytes `b"not a video"`): `figure 1 light-content: video cannot be decoded:`.
    - `test_an_encoded_nul_in_a_video_path` (dark `src="../media/demo_2d.dark%00.mp4"`): `video path contains a NUL`.
    - `test_a_video_outside_the_site` (light `src="/media/demo_2d.light.mp4"`): `video is outside the site`.
    - `test_no_archive_for_the_clip` (`ARCHIVES` pointed at an empty directory): `figure 1: no readable archive for clip 'demo_2d'`.
    - `test_no_encoder` (`check_site_figures.find_encoder` raising `EncoderNotFound("none")`): exactly `["clips cannot be decoded: none"]`.
    - `test_the_responsive_rule_is_required` (`styles.css` emptied): exactly one failure, `styles.css: the rule 'figure.replay video { max-width: 100%; height: auto; }' is missing`; and `test_the_responsive_rule_may_be_minified` (`styles.css` holding `figure.replay video{max-width:100%;height:auto;}`): `[]`.
    - `test_plot_pages_need_no_stylesheet`: the existing `_build(tmp_path)` with no `styles.css` still gives `[]`.
  - Deferred from pull request 2: `test_pages_under_site_libs_are_not_scanned` (`site_libs/x/y.html` holding a light wrapper, the manifest page well formed): `[]`; and `test_an_unknown_kind_is_a_programming_error` (manifest `{PAGE: (Fig("map", 420),)}` on a well-formed plot page): `pytest.raises(ValueError, match="figure 1: no check for kind 'map'")`.

  Each negative fixture is the only test that fails when its one check is deleted; the mutation table of Task 16 names them check by check.

- [ ] **Step 2: Stage, then run.** Add `ARCHIVES`, `RESPONSIVE_RULE`, the `what` parameter, and to `_FigureParser` the three new collections left empty; import the `clocks._video` names. Run `uv run pytest tests/test_check_site_figures.py -q`: every clip test fails: where the page is otherwise well formed, with today's `ValueError: ... no check for kind 'clip'` (two themed videos are two plot wrappers today), and elsewhere on its assertion; the two deferred tests pass (they pin present behaviour), and every existing test passes.

- [ ] **Step 3: Implement** as "What the check does with a clip" says. Update the module docstring: clips, what is decoded, and that the stylesheet and encoder checks apply only when a clip is listed.

- [ ] **Step 4: Tests pass; full gate; render the site and run both checks** (the checker changed): `Figure contract OK: 9 pages, 11 figures checked.`

- [ ] **Step 5: Commit and push** as `Teach the rendered-site check clips: markup, files, posters and the decoded stream`.

---

### Task 8: The pre-render step

After this task a site build draws the twelve clips and posters into `site/media/` and copies them to the output; no page embeds them until Task 10.

**Files:**
- Create: `src/clocks/_site_media.py`, `site/prerender.py`, `tests/test_site_media.py`
- Modify: `site/_quarto.yml`, `site/styles.css`, `.gitignore`

**Interfaces:**
- Consumes: `CLIP_HEIGHTS`, `render_clip`, `final_frame` from `clocks._clips`; `find_encoder`, `encoder_version` from `clocks._video`; `load_replay`, `DEMOS` from `clocks._replay`; `style`, `save_png`, `THEMES` from `clocks._style`.
- Produces, in `src/clocks/_site_media.py`:

```python
CLIP_DEMOS: tuple[str, ...]  # DEMOS without "demo_density", in DEMOS order
PACKAGE: Path                # the directory of the installed clocks package

def media_paths(media: Path, demo: str, theme: str) -> tuple[Path, Path]: ...
def render_site_media(archives: Path, media: Path, *, full: bool,
                      demos: Sequence[str] = CLIP_DEMOS, sources: Path = PACKAGE,
                      report: Callable[[str], None] = print) -> list[Path]: ...
```

  `media_paths` returns `(media / f"{demo}.{theme}.mp4", media / f"{demo}.{theme}.png")`. `render_site_media`, in this order:
  1. finds the encoder; `EncoderNotFound` propagates before anything is created, written or removed. Reports `f"encoder: {encoder} ({encoder_version(encoder)})"`.
  2. creates `media` if missing.
  3. for each demo, then each theme of `THEMES`: the pair is *stale* when `full` is true, or either file is missing, or the older of the two is not strictly newer than both `archives / f"{demo}.npz"` and the newest regular file under `sources` (directories named `__pycache__` skipped). A stale pair is rewritten: the clip by `render_clip(replay, clip_path, theme=theme, encoder=encoder)`, then the poster by `save_png(final_frame(replay, colors), temporary)` inside `with style(theme) as colors:`, the temporary file being `.<poster name>.<random>.tmp` in `media`, moved over the poster with `os.replace`, and the figure closed in a `finally`. Reports `f"rendered {clip_path.name} ({frames} frames, {seconds:.1f} s)"`, or `f"kept {clip_path.name}"`.
  4. when `full`, removes every regular file directly inside `media` whose suffix is `.mp4`, `.png` or `.tmp` and which is not one of the files of step 3, reporting `f"removed {name}"`. It never enters a subdirectory and never removes a file with another suffix.
  5. returns the paths it wrote, clip then poster, in the order written.
- `site/prerender.py`, whole:

```python
"""Quarto pre-render step: draw the site's clips and posters into media/."""

import os
from pathlib import Path

from clocks._site_media import render_site_media

SITE = Path(__file__).resolve().parent
render_site_media(
    SITE.parent / "assets" / "replays",
    SITE / "media",
    full=os.environ.get("QUARTO_PROJECT_RENDER_ALL") == "1",
)
```

- `site/_quarto.yml`: under `project:`, after `output-dir`, the lines `  pre-render: prerender.py` and `  resources:` with the two entries `    - "media/*.mp4"` and `    - "media/*.png"`. `site/styles.css` gains the rule of Global Constraints, with a comment that a clip is declared 750 pixels wide and must shrink with a narrow column. `.gitignore` gains `site/media/` under "Quarto build products".

- [ ] **Step 1: Write the failing tests** in `tests/test_site_media.py`. A fixture `archives` holds `demo_2d.npz`, `save_replay` of `demo_2d.record(n_observations=2, n_particles=16)`; `sources` is a directory holding one file `a.py`; every call passes `demos=("demo_2d",)`, `sources=sources` and a `report` that appends to a list. "Times set" means `os.utime` on the archive, the source file and the four outputs.

  - `test_a_full_render_writes_each_clip_and_poster`: the result is `[demo_2d.light.mp4, demo_2d.light.png, demo_2d.dark.mp4, demo_2d.dark.png]` under `media`, which holds exactly those; each clip's `probe` is `StreamInfo("h264", "yuv420p", 1500, 900, 20.0, 92, True)`; each poster's `png_size` is (1500, 900) and its first pixel that theme's background. Only this catches a poster in the wrong theme or a pair written under the wrong names.
  - `test_a_full_render_removes_other_media_and_nothing_else`: with `old.mp4`, `old.png`, `.x.tmp`, `notes.txt` and `sub/keep.mp4` in `media` beforehand, a full render leaves the four outputs, `notes.txt` and `sub/keep.mp4`, and reports `removed old.mp4`, `removed old.png` and `removed .x.tmp`. Only this catches a stale clip surviving a full render, or a cleanup that deletes what it does not own (decision D14).
  - `test_a_partial_render_keeps_fresh_media`: after a full render, times set to 1,000 for the archive and source and 2,000 for the outputs; `full=False` returns `[]`, reports `kept demo_2d.light.mp4` and `kept demo_2d.dark.mp4`, and `clocks._site_media.render_clip`, wrapped by a spy, is not called. Only this catches a preview that redraws everything.
  - `test_a_partial_render_rebuilds_what_is_stale` (IDs and the change after the same setup): `archive_newer` (archive at 3,000) → all four rewritten; `source_newer` (the source file at 3,000) → all four; `tie` (archive at 2,000, the outputs' own time) → all four (decision D15); `poster_missing` (the light poster deleted) → the two light files only; `clip_older` (the light clip at 500) → the two light files only; `stray_file_kept` (`media/old.mp4` added, nothing stale) → `[]`, and `old.mp4` still there. Each asserts the returned paths. Only this catches, case by case, a staleness rule that ignores the archive, the sources, a missing file or the older of the two, and a partial render that deletes.
  - `test_a_cached_bytecode_file_does_not_make_media_stale`: `sources / "__pycache__" / "a.pyc"` at 3,000 with everything else fresh → `[]`. Only this catches every test run re-rendering the site's media.
  - `test_no_encoder_stops_before_anything_is_touched`: `clocks._site_media.find_encoder` raises `EncoderNotFound("none")`; with `media` absent, a full render raises it and `media` still does not exist; with `media/old.mp4` present, it is still there. Only this catches a full render that clears the directory and then cannot refill it.
  - `test_the_poster_is_the_last_row_of_its_archive`: three full renders into three directories in one process: from the fixture archive; from a copy with 0.5 added to `cloud[-1]`; from a copy with 0.5 added to `cloud[1]`. Read with `matplotlib.image.imread`, the light poster of the second differs from the first's, and the third's equals the first's (`np.array_equal`). Only this catches a poster drawn from another row, or media rendered from another archive than the one given.
  - `test_a_failed_poster_leaves_the_pair_stale`: `clocks._site_media.save_png` is replaced by a stand-in that, on its first call, writes `b"partial"` to the path it is given and raises `OSError("disk")`, and afterwards calls through; a full render raises that error; `media` holds `demo_2d.light.mp4` and no `.png` and no `.tmp` file; a following partial render, with the fault removed, returns all four paths. Only this catches a half-written poster left under its final name or a temporary file left behind.
  - `test_the_report_names_the_encoder_and_each_file`: the first report line starts `encoder: ` and contains `ffmpeg version`; the next two match `rendered demo_2d\.(light|dark)\.mp4 \(92 frames, \d+\.\d s\)`. Only this catches a build log that cannot say which encoder ran.
  - `test_site_media_covers_the_clip_demos`: `CLIP_DEMOS == tuple(name for name in DEMOS if name != "demo_density")` and `set(CLIP_DEMOS) == set(CLIP_HEIGHTS)`.
  - `test_the_prerender_script_is_its_pinned_program`: `ast.dump(ast.parse(<site/prerender.py>)) == ast.dump(ast.parse(PINNED))`, `PINNED` being the program above. Only this catches the script pointed at another directory or always passing `full=True`.
  - `test_quarto_runs_the_script_and_ships_the_media`: `site/_quarto.yml`, as text, has `  pre-render: prerender.py` and, after a line `  resources:`, the two entries, all before the first line that starts a new top-level key; `.gitignore` has the line `site/media/`; `site/styles.css` with whitespace removed contains `figure.replayvideo{max-width:100%;height:auto;}`. Only this catches media drawn but never copied to the site, or committed by accident.

- [ ] **Step 2: Stage the names, then run.** Create `_site_media.py` with `CLIP_DEMOS`, `PACKAGE`, `media_paths` and a `render_site_media` that returns `[]`; create `site/prerender.py` as above. Run `uv run pytest tests/test_site_media.py -q`: the render tests fail on their assertions, the configuration test on the missing lines; the pinned-program and demo-list tests pass.

- [ ] **Step 3: Implement** as Interfaces says. `tests/test_site_figure_style.py`'s `test_clip_modules_import_no_model_code` (Task 3) now covers `_site_media.py`.

- [ ] **Step 4: Tests pass; full gate; render the site and run both checks.** Delete `site/_output` and `site/media` first, so the build is a clean one. Expected: the render's log shows the `encoder:` line and twelve `rendered` lines; `site/_output/media/` holds 24 files; `check_site_figures.py` prints `Figure contract OK: 9 pages, 11 figures checked.` Then run `cd site && uv run --frozen quarto render index.qmd && cd ..` and confirm its log shows twelve `kept` lines. Record the twelve times and the two renders' wall-clock times for the pull request.

- [ ] **Step 5: Commit and push** as `Draw the site's clips and posters in a Quarto pre-render step`. The body gives the local Quarto version and the render times.

---

### Task 9: The source guard learns replay embeds; two notes owed from pull request 2

Helpers and their tests only. The table of real embeds is empty until Task 10, so every page still passes.

**Files:**
- Modify: `tests/test_site_figure_style.py`

**Interfaces:**
- Consumes: `CLIP_HEIGHTS` from `clocks._clips`; `Fig`, `EXPECTED_FIGURES` from the script, as today.
- Produces, in `tests/test_site_figure_style.py`:

```python
class Embed(NamedTuple):
    clip: str
    label: str    # the aria-label of both videos
    caption: str

REPLAY_EMBEDS: dict[str, tuple[Embed, ...]] = {}  # page, relative to site/, to its embeds in order
FIGURE_PAGES: tuple[str, ...]                      # sorted(set(PLOT_PAGES) | set(REPLAY_EMBEDS))

def _description_id(clip: str) -> str: ...        # "demo_2d" -> "demo-2d-text"
def _embed_block(embed: Embed, page: str) -> str: ...
def _replay_embed_violations(text: str, page: str, embeds: Sequence[Embed]) -> list[str]: ...
def _descriptions(text: str) -> dict[str, str]: ...
def _page_figures(text: str, page: str) -> list[tuple[str, int, str | None]]: ...
def _without(description: str, sentences: Sequence[str], allowed: Sequence[str] = ()) -> str: ...
```

  - `_embed_block(embed, page)` returns exactly this text, with `{id}` from `_description_id`, `{h}` from `CLIP_HEIGHTS[embed.clip]` and `{up}` being `"../"` once per directory level of `page` (`""` for `index.qmd`):

````text
```{=html}
<figure class="replay">
<video class="light-content" aria-label="{label}" aria-describedby="{id}" src="{up}media/{clip}.light.mp4" poster="{up}media/{clip}.light.png" width="750" height="{h}" controls playsinline preload="none"></video>
<video class="dark-content" aria-label="{label}" aria-describedby="{id}" src="{up}media/{clip}.dark.mp4" poster="{up}media/{clip}.dark.png" width="750" height="{h}" controls playsinline preload="none"></video>
<figcaption>{caption}</figcaption>
</figure>
```
````

  - `_replay_embed_violations(text, page, embeds)`: for embed number n (from 1), `f"embed {n} ({clip}): the pinned block is missing or altered"` when `_embed_block` does not occur in `text`, or `f"embed {n} ({clip}): the pinned block occurs {k} times"`; `"the embeds are out of order"` when all occur once but not in the table's order; for each embed found once, `f"embed {n} ({clip}): not followed by its description `::: {{#{id}}}`"` unless the next non-blank line after the block is exactly `::: {#<id>}`, or `f"embed {n} ({clip}): the description is empty"` when no non-blank line lies between that line and the closing `:::`, or `f"embed {n} ({clip}): the description id occurs {k} times"` when `{#<id>}` occurs more than once in `text`; and, in the text with every pinned block removed, one `f"unpinned video markup: line {m}"` for each line (numbered in the original text) that contains, in any letter case, `<video`, `<source`, `<figure`, `<iframe`, `.mp4` or `{=html}`.
  - `_descriptions(text)` maps the id of each `::: {#<id>}` div to its body, whitespace-normalised.
  - `_page_figures(text, page)` returns, in source order, `("plot", _cell_height(cell), None)` for each plot cell and `("clip", CLIP_HEIGHTS[clip], clip)` for each of the page's pinned blocks.
  - `_without(description, sentences, allowed)` returns the description with each of `sentences` and then each of `allowed` removed; it raises `AssertionError` naming the first sentence that is not in the description.

- [ ] **Step 1: Write the failing tests.**

  - `test_embed_block_is_the_pinned_markup`: `_embed_block(Embed("demo_2d", "Animation: the 2D search", "A caption."), "story/into-the-plane.qmd")` equals the block above with `Animation: the 2D search`, `demo-2d-text`, `../`, `450` and `A caption.` written out in the test as one literal; and for `"index.qmd"` the four paths start `media/`. Only this catches a generated block that drifts from the design's markup: an attribute dropped, reordered or renamed.
  - `test_replay_embed_violations`, on a synthetic page `PAGE_TEXT`: a paragraph, the block for `demo_2d`, a blank line, `::: {#demo-2d-text}`, `Words.`, `:::`. IDs and expectations:
    - `canonical`: `[]`;
    - `autoplay_added` (` autoplay` inserted before `></video>` on the light line): the first message is `embed 1 (demo_2d): the pinned block is missing or altered`, and every other message starts `unpinned video markup: line `;
    - `caption_changed`, `height_changed` (`height="500"` on both), `dark_video_removed`, `fence_removed` (the two fence lines deleted): the same first message;
    - `block_twice`: `embed 1 (demo_2d): the pinned block occurs 2 times`;
    - `a_second_video` (`<video src="x.mp4"></video>` on a later line): exactly `[f"unpinned video markup: line {n}"]`;
    - `a_link_to_a_clip` (`[the clip](../media/demo_2d.light.mp4)` in the paragraph): the same message for that line;
    - `an_iframe`, `a_source_element`, `upper_case_video` (`<VIDEO ...>`), `another_raw_block` (a `{=html}` block holding a `<p>`): one `unpinned video markup` message each;
    - `description_missing` (the div removed), `description_not_adjacent` (a paragraph between block and div): `embed 1 (demo_2d): not followed by its description `::: {#demo-2d-text}``;
    - `description_empty` (the div's body blank): `embed 1 (demo_2d): the description is empty`;
    - `description_id_twice` (a second div with the id, later): `embed 1 (demo_2d): the description id occurs 2 times`;
    - `two_embeds_swapped` (the table lists `demo_multi_mass` then `demo_multi_mass_2d`, the page has them the other way round, each with its div): `["the embeds are out of order"]`.
    Only this catches, case by case, an embed edited by hand, a second video outside the table, and a description that the videos name but that is not there.
  - `test_descriptions_are_found_by_id`: on `PAGE_TEXT` with a second div `::: {#other}` and a callout `::: {.callout-note}`: `{"demo-2d-text": "Words.", "other": ...}`, the callout absent.
  - `test_page_figures_merges_cells_and_embeds_in_source_order`: a synthetic page with a plot cell holding `figsize=(7.5, 5.4)`, then the block, then a plot cell without it, checked against a one-entry table: `[("plot", 540, None), ("clip", 450, "demo_2d"), ("plot", 420, None)]`. Only this catches clips listed after all plots whatever their place on the page.
  - `test_without_removes_pinned_sentences_and_reports_a_missing_one`: `_without("A 1. B 2. mass 1.", ["A 1."], ["mass 1"])` is `" B 2. ."`; a sentence that is absent raises `AssertionError` naming it.
  - The real pages, vacuous until Task 10: `test_no_page_has_unpinned_video_markup` (every `.qmd` of `_MENTION_FILES`, ID the path): `_replay_embed_violations(text, page, REPLAY_EMBEDS.get(page, ())) == []`; `test_embed_labels_and_captions_are_plain_text`: no label or caption in `REPLAY_EMBEDS` contains any of `<`, `>`, `&`, `"`.
  - `test_manifest_matches_the_source` is rewritten over `FIGURE_PAGES`: `tuple(EXPECTED_FIGURES[_manifest_page(page)])` as `(kind, height, clip)` triples equals `tuple(_page_figures(text, page))`. The assertion `fig.clip is None` goes: a plot's `clip` is `None` and a clip's is its name, by the comparison itself.
  - `test_the_manifest_lists_exactly_the_plot_pages` becomes `test_the_manifest_lists_exactly_the_figure_pages`: the manifest's keys equal `FIGURE_PAGES` as `.html`; `len(PLOT_PAGES) == 9` stays.
  - Owed from pull request 2, Codex round 6. The `spaced_marker` case of `test_builder_cell_options_are_exactly_the_three` puts its marker after a statement, where Quarto ignores it: build it as `_CANONICAL_OPTIONS + "# | classes: [d-none]\n" + _SMALL_PROGRAM`, directly under the five-line header, with the same expected message, `unexpected comment line: '# | classes: [d-none]'`. And the docstring of `test_builder_cell_is_exactly_its_pinned_program` says "comments and spacing do not matter", which the header and comment checks made untrue: it becomes "The program is compared by syntax tree, so spacing inside a statement does not matter and every statement, name, literal and argument does; the option block and every comment line are compared as text, by `_builder_cell_option_violations`."
  - The module docstring gains: "Clip embeds are pinned whole: each is the text `_embed_block` builds from `REPLAY_EMBEDS`, and video markup anywhere else on a page is a violation. This catches an embed changed by mistake. It does not prove that a clip is visible or plays on the rendered page; the rendered-site check decodes the files and checks the markup that reached the page, and the browser checks cover playback."

- [ ] **Step 2: Stage the helpers, then run.** Add `Embed`, the empty table, `FIGURE_PAGES`, `_description_id`, and the other helpers returning `""`, `[]`, `{}`, `[]` and the description unchanged. Run `uv run pytest tests/test_site_figure_style.py -q`: the helper tests fail on their assertions; the rewritten manifest test fails (`_page_figures` returns nothing); the moved `spaced_marker` case passes, as it did.

- [ ] **Step 3: Implement** the helpers.

- [ ] **Step 4: Tests pass; full gate; commit and push** as `Pin clip embeds whole in the page-source guard; settle two notes from the figures review`.

---

### Task 10: The switch: seven embeds, six descriptions, and the GIFs retired

One change, so that every sentence is true at its commit: the pages show the clips, the manifest lists them, the GIFs and everything that names them go, and the provenance prose describes what is there.

**Files:**
- Modify: `site/index.qmd`, `site/story/the-search-in-one-dimension.qmd`, `site/story/into-the-plane.qmd`, `site/story/two-hidden-masses.qmd`, `site/story/how-many-masses.qmd`, `site/story/gravitational-echolocation.qmd`, `site/reproduce/reproducibility.qmd`, `README.md`, `docs/2026-08-16-development-calibration.md`, `scripts/check_site_figures.py`
- Delete: the six `demo_*.gif` under `assets/` and the same six under `site/assets/`
- Test: `tests/test_site_figure_style.py`, `tests/test_generated_assets.py`

**Interfaces:**
- Consumes: Task 9's helpers; `load_replay`; `trace_decimals` from `clocks._figures`.
- Produces: `REPLAY_EMBEDS`, as below; `EXPECTED_FIGURES`, as below; in `tests/test_generated_assets.py`, `RETIRED_VISUALS` extended by the six GIFs and `EXPECTED_VISUAL_HASHES` deleted with its test.

```python
REPLAY_EMBEDS = {
    "index.qmd": (Embed(
        "demo_2d", "Animation: the 2D search",
        "One simulated run: a particle filter finding a hidden mass on a plane from "
        "clock rates alone. Left: the cloud of hypotheses, the eight clocks and the "
        "true position. Right: the estimates of x, y and M after each observation."),),
    "story/the-search-in-one-dimension.qmd": (Embed(
        "demo_1d", "Animation: the search in one dimension",
        "The 1D search, one simulated run. Left: the cloud of hypotheses in position "
        "and mass, above the three clocks and, clock by clock, the true, predicted "
        "and measured slowing. Right: the estimates of x and M after each observation."),),
    "story/into-the-plane.qmd": (Embed(
        "demo_2d", "Animation: the 2D search",
        "The 2D search, one simulated run: 8 clocks, one hidden mass, three unknowns "
        "(x, y, M). Left: the cloud and the clocks, filled darker the slower they "
        "run. Right: the three estimates after each observation."),),
    "story/two-hidden-masses.qmd": (
        Embed("demo_multi_mass", "Animation: two masses on a line",
              "Two masses in 1D: five clocks, four unknowns. One simulated run."),
        Embed("demo_multi_mass_2d", "Animation: two masses on a plane",
              "Two masses in 2D: ten randomly placed clocks, six unknowns. "
              "One simulated run."),
    ),
    "story/how-many-masses.qmd": (Embed(
        "demo_model_comparison",
        "Animation: the probability of one, two and three masses",
        "One simulated run: posterior probability over K=1, 2, 3 as 80 observations "
        "accumulate. Truth is K=2."),),
    "story/gravitational-echolocation.qmd": (Embed(
        "demo_echolocation_3d", "Animation: the head locating an exterior mass",
        "The echolocation demo, one simulated run. Left: the head, its clocks, the "
        "cloud and the true mass (ring), with the whole search box at upper left. "
        "Right: the four estimates after each observation, and the fitted readings "
        "against the observed ones."),),
}

EXPECTED_FIGURES = {
    "index.html": (Fig("clip", 450, "demo_2d"),),
    "story/clocks-as-gravimeters.html": (Fig("plot", 420),),
    "story/one-clock-is-not-enough.html": (Fig("plot", 420),),
    "story/the-search-in-one-dimension.html": (Fig("clip", 450, "demo_1d"), Fig("plot", 540)),
    "story/into-the-plane.html": (Fig("clip", 450, "demo_2d"), Fig("plot", 540)),
    "story/two-hidden-masses.html": (
        Fig("clip", 500, "demo_multi_mass"), Fig("clip", 550, "demo_multi_mass_2d"),
        Fig("plot", 420),
    ),
    "story/how-many-masses.html": (Fig("clip", 420, "demo_model_comparison"), Fig("plot", 420)),
    "story/beyond-point-masses.html": (Fig("plot", 540), Fig("plot", 420)),
    "story/gravitational-echolocation.html": (
        Fig("plot", 420), Fig("clip", 550, "demo_echolocation_3d"), Fig("plot", 540),
    ),
    "method/the-particle-filter.html": (Fig("plot", 540),),
}
```

  The captions' layout words ("Left", "Right", "at upper left") follow the design's arrangement; if Task 1's layout differs, the caption follows the layout and Task 13 checks it by eye.

**The six descriptions.** Each goes in a `::: {#<id>}` div directly under its block. A number is written with `trace_decimals` of its parameter's prior range (two decimals for positions, three for masses), a minus sign as the ASCII hyphen. The text is given here with today's numbers (Q8); the tests below build the numbered sentences from the archives.

  - `demo-1d-text`: "Before any reading the dots cover the whole search: every position along the track and every mass the prior allows. The first reading pulls them into a narrow band around the true position, and each of the next few tightens it; under the cloud, the filter's predicted slowing at each clock closes on the measured one. After 50 readings the run ends at x = 2.48 ± 0.04 and M = 0.101 ± 0.001, against a true x = 2.50 and M = 0.100. Each ± is one standard deviation of the final cloud: how far the surviving hypotheses still disagree in this one run, not an error bar over many runs."
  - `demo-2d-text`, the same words on the landing page and on "Into the Plane": "Before any reading the dots cover the whole square. The first readings pull them toward the mass, and by the eighth they are a small cloud around it; the three traces at the right narrow at the same time. After 50 readings the run ends at x = 1.43 ± 0.07, y = -1.15 ± 0.08 and M = 0.149 ± 0.001, against a true x = 1.50, y = -1.00 and M = 0.150. Each ± is one standard deviation of the final cloud. The y estimate sits about two of those from the truth: that is this run's noise, and another seed would miss in another direction."
  - `demo-multi-mass-text`: "Before any reading both clouds are spread along the whole track. The first mass is narrowed at once; the second takes longer: after one reading its position is x₂ = 6.61 ± 1.04, and after eight it is 4.62 ± 0.18. After 80 readings the run ends at x₁ = -2.00 ± 0.04, x₂ = 4.55 ± 0.05, M₁ = 0.045 ± 0.001 and M₂ = 0.030 ± 0.001, against true values of -2.00, 4.50, 0.045 and 0.030. Each ± is one standard deviation of the final cloud for that parameter."
  - `demo-multi-mass-2d-text`: "Before any reading both clouds cover the whole square. The first readings barely narrow them, and for a while each cloud is split between two arrangements of the masses, which the ellipses cannot show and the dots can. After 80 readings the run ends at x₁ = -2.38 ± 0.42, y₁ = 2.03 ± 0.22, x₂ = 4.13 ± 0.80, y₂ = -0.95 ± 0.63, M₁ = 0.052 ± 0.004 and M₂ = 0.028 ± 0.003, against true values of -3.00, 2.00, 4.00, -1.00, 0.050 and 0.030. The final clouds are still wide: this run locates both masses loosely, and each ± is one standard deviation of its final cloud." The clause about two arrangements is the design's reading of this run ("the cloud passes through a second mode"); it is believed, not measured, and Task 13 removes it if the sheets do not show it.
  - `demo-model-comparison-text`: the paragraph already under the figure, unchanged, wrapped in the div: "One mass is all but ruled out by the first observation. Three masses lead at first, two take over around the tenth observation, and from about the twentieth three stay ahead: the run ends at about 40% for two masses and 60% for three, leaning toward the wrong answer. That is one run's estimate from finite data and a finite particle cloud, a weak preference rather than a verdict."
  - `demo-echolocation-3d-text`: "Before any reading the dots fill the whole search box, 32 units on a side, and only 0.4% of the prior lies inside the close view around the head. The first reading collapses the cloud into that view, where 100.0% of its weight stays; the small panel of the whole box shows one cluster where the box was full. After 80 readings the run ends at x = 0.98 ± 0.02, y = 1.46 ± 0.02, z = 2.93 ± 0.03 and M = 0.078 ± 0.002, against true values of 0.99, 1.48, 2.97 and 0.080. Each ± is one standard deviation of the final cloud. The panel of fitted against observed readings shows agreement with the readings used in the fit, not a test on new data."

- [ ] **Step 1: Write the failing tests.**

  In `tests/test_site_figure_style.py`:

  - Fill `REPLAY_EMBEDS`; `RETIRED_ASSETS` gains the six GIF names; `test_the_manifest_lists_exactly_the_figure_pages` also asserts `len(FIGURE_PAGES) == 10`. `test_manifest_matches_the_source`, `test_no_page_has_unpinned_video_markup` and `test_no_page_mentions_a_retired_asset` now fail until the pages and the manifest change.
  - `test_each_clip_is_embedded_at_its_declared_height`: for every embed, the manifest's `Fig` for it has `height == CLIP_HEIGHTS[clip]`; and the seven embeds name each of the six clip demos at least once. Only this catches a manifest height typed by hand that the drawn clip does not have.
  - `test_description_numbers_are_the_archives` (one case per embed, ID `<page>:<clip>`). With `d = _descriptions(text)[_description_id(clip)]`, `replay = load_replay(REPO / "assets/replays" / f"{clip}.npz")`, `m = mean[-1]`, `s` the last row's standard deviations, `t = true_params`, `T = n_observations`, and `f(j, v)` formatting `v` with `trace_decimals(prior_ranges[j])` decimals, the case builds its sentences and asserts `re.search(r"\d", _without(d, sentences, allowed)) is None`:
    - `demo_1d`: `f"After {T} readings the run ends at x = {f(0, m[0])} ± {f(0, s[0])} and M = {f(1, m[1])} ± {f(1, s[1])}, against a true x = {f(0, t[0])} and M = {f(1, t[1])}."`; and, for the words "narrow band", `sd[1, 0] < 0.1 * sd[0, 0]` (0.27 against 4.92 today).
    - `demo_2d` (both pages): `f"After {T} readings the run ends at x = {…} ± {…}, y = {…} ± {…} and M = {…} ± {…}, against a true x = {…}, y = {…} and M = {…}."` in the same pattern; for "by the eighth", `sd[8, :2].max() < 0.1 * sd[0, :2].min()`; for "about two of those", `1.5 < abs(m[1] - t[1]) / s[1] < 2.5`.
    - `demo_multi_mass`: `f"after one reading its position is x₂ = {f(1, mean[1, 1])} ± {f(1, sd[1, 1])}, and after eight it is {f(1, mean[8, 1])} ± {f(1, sd[8, 1])}."` and `f"After {T} readings the run ends at x₁ = {…} ± {…}, x₂ = {…} ± {…}, M₁ = {…} ± {…} and M₂ = {…} ± {…}, against true values of {…}, {…}, {…} and {…}."`; for "takes longer", `sd[1, 1] > 1.5 * sd[1, 0]` (1.04 against 0.49 today).
    - `demo_multi_mass_2d`: the final sentence in the same pattern with six parameters; for "barely narrow", every position standard deviation at row 3 exceeds 0.4 times its row-0 value.
    - `demo_model_comparison`: with `leader[k]` the candidate of largest `model_posterior[k]` for k ≥ 1 and `changes` the observations at which it changes, the test asserts `leader[1] == 3`, `changes == [10, 19]` with leaders 2 then 3, and that the description contains `f"Three masses lead at first, two take over around the {ORDINAL[round(changes[0], -1)]} observation, and from about the {ORDINAL[round(changes[1], -1)]} three stay ahead: the run ends at about {round(100 * p2)}% for two masses and {round(100 * p3)}% for three"`, with `ORDINAL = {10: "tenth", 20: "twentieth"}` and `p2`, `p3` the final probabilities looked up by K in `model_ks`; and `model_posterior[1, 0] < 1e-5` for "all but ruled out by the first observation".
    - `demo_echolocation_3d`: `f"Before any reading the dots fill the whole search box, {hi - lo:g} units on a side, and only {100 * w[0]:.1f}% of the prior lies inside the close view around the head."` with `(lo, hi)` the prior range of x; `f"where {100 * w[1:].min():.1f}% of its weight stays"`; the final sentence with four parameters, true values `f(j, t[j])`.
    `allowed` is `("mass 1", "mass 2")`. Only this catches a description whose numbers belong to another run, or a number added to a description without a check.
  - `test_the_two_2d_descriptions_are_the_same`: `_descriptions` of the landing page and of "Into the Plane" give equal text under `demo-2d-text` (decision D24). Only this catches one of the two being edited alone.
  - `test_the_model_caption_counts_the_archives_observations`: the caption of the `demo_model_comparison` embed contains `f"as {T} observations accumulate"` and `f"Truth is K={true_k}"`, from the archive's `meta`.
  - `test_the_1d_page_says_how_many_dots_are_shown`: the archive has `n_particles == 1000` and `cloud_size == 500`, and the page, whitespace-normalised, contains "one thousand of them in the run animated above, which shows a sample of five hundred". Only this catches the page claiming the clip shows the whole cloud.

  In `tests/test_generated_assets.py`:

  - Delete `EXPECTED_VISUAL_HASHES`, `test_corrected_visual_release_bytes_are_complete_and_mirrored` and the loop over its hashes in `test_current_prose_identifies_corrected_assets_and_honest_model_results`. `RETIRED_VISUALS` gains the six GIF names with the hashes the deleted table held; `test_retired_visuals_are_gone_but_recorded` then covers eight files.
  - In that prose test, the two GIF sentences become: `f"The animation on [How Many Masses?](../story/how-many-masses.qmd) replays the packaged default run: {T} observations, {N:,} particles per candidate, and seed {seed}. "` and `f"With K={true_k} truth, its final estimate is K=2: {p2:.4f} and K=3: {p3:.4f}. "`, with the values from `assets/replays/demo_model_comparison.npz` (`p2`, `p3` looked up by K in `model_ks`); the 25-observation and horizon sentences stay as they are.
  - `test_density_provenance_names_its_run` pins the two sentences that replace its one: `"**Replay data.** Each animation, and the density figure on [Beyond Point Masses](../story/beyond-point-masses.qmd), is drawn when the site is built, from `assets/replays/<name>.npz`, the recorder's archive of that demo's packaged run. "` and `f"For the density demo that is {n} observations, {N:,} particles and seed {seed}. "`.
  - `test_the_cloud_size_in_the_prose_is_the_archives` also asserts that every archive with a `cloud` has `cloud_size == 500`, and that the page contains "except for the model-comparison run, a 500-particle sample for display".
  - `test_deploy_sentence_says_how_each_figure_is_redrawn` pins: "so every figure and animation is redrawn at deploy time: most figures by running the model again from source, and the density figure, the range-study figure and the six animations from committed data files." and asserts that neither "demo GIFs" nor "committed images" is on the page.
  - `test_video_files_are_called_build_products`: the page contains "The video files are build products: they are not committed, and their bytes are not pinned."
  - `test_echolocation_animation_provenance`: the page contains "**Echolocation animation.** The animation places the mass at two circumradii along a fixed off-axis direction, and is one fixed-seed visualization at the closest, most favorable range, not a reliability measurement." and none of "will change when the assets are next regenerated", "not a re-render of the current code", "committed GIF".
  - `test_readme_points_to_the_site_and_describes_replay_data`: the README, whitespace-normalised, contains "The repository commits no rendered media." and "It commits replay data, `assets/replays/<demo>.npz`: what each packaged run produced, in numbers, tied to the code by tests of its inputs and of bounded summaries.", and does not contain `assets/demo_2d.gif`.

- [ ] **Step 2: Run** `uv run pytest tests/test_site_figure_style.py tests/test_generated_assets.py tests/test_check_site_figures.py -q`: the embed, manifest, retired-asset, description and prose tests fail; nothing errors on a missing name.

- [ ] **Step 3: Implement.**

  - Each page: replace the Markdown image line by `_embed_block(...)`'s text for that page (generate it with the helper and paste it; the test compares the whole string), a blank line, and the description div. On "How Many Masses?" the div wraps the existing paragraph. Three sentences that describe an animation's layout change with it: on "The Search in One Dimension", "one thousand of them in the animation above" becomes "one thousand of them in the run animated above, which shows a sample of five hundred"; on "Gravitational Echolocation", "The animation's rate panel subtracts the channel mean from each clock's rate, while" becomes "The animation's panel of fitted against observed readings uses readings with the channel mean subtracted from each clock's rate, while", and "while the camera orbits" becomes "while the camera turns part of the way around the head".
  - `scripts/check_site_figures.py`: the manifest above; in the docstring, "Images outside the wrappers (committed GIFs) are ignored" becomes "Images outside the wrappers are ignored".
  - `git rm` the twelve GIFs.
  - `docs/2026-08-16-development-calibration.md`: after the two sentences pull request 2 added, one sentence per GIF, with today's date in digits: "`demo_1d.gif` was removed from both trees on YYYY-MM-DD; the animation is now drawn when the site is built, from `assets/replays/demo_1d.npz`, and the GIF's old bytes remain in git history under the hash above." The table is not edited.
  - `site/reproduce/reproducibility.qmd`:
    - the `site.yml` bullet ends "…so every figure and animation is redrawn at deploy time: most figures by running the model again from source, and the density figure, the range-study figure and the six animations from committed data files."
    - the paragraphs "**Demo animations.**" and "**Replay data.**" become one: "**Replay data.** Each animation, and the density figure on [Beyond Point Masses](../story/beyond-point-masses.qmd), is drawn when the site is built, from `assets/replays/<name>.npz`, the recorder's archive of that demo's packaged run. For the density demo that is 60 observations, 2,000 particles and seed 42. An archive holds the configuration, seed and observations, a summary of the cloud before any observation and after each, and, except for the model-comparison run, a 500-particle sample for display. Tests run each demo again; they compare its inputs, the computed ones (clock positions, true rates and readings) at `rtol=1e-12, atol=1e-12` and the rest exactly, and its summaries within stated bounds; they do not require the same random path. The video files are build products: they are not committed, and their bytes are not pinned. `uv run demo-density` records the same configured scenario and draws its result. Each archive is one finite realization: a fixed scenario and seed. None establishes a population recovery rate. Where a story page also runs a smaller example in an executed cell, that is a separate finite run whose settings are shown in the cell."
    - "**Echolocation animation.**" keeps only the sentence the test pins.
    - "**Model-comparison runs.**" opens with the two sentences the test pins; the rest of the paragraph is unchanged.
  - `README.md`: the image line and the paragraph that begins "The committed demo media" become: "The site shows each demo as an animation: [jbwhit.github.io/clocks](https://jbwhit.github.io/clocks/). The repository commits no rendered media. It commits replay data, `assets/replays/<demo>.npz`: what each packaged run produced, in numbers, tied to the code by tests of its inputs and of bounded summaries. The site draws its animations and the density figure from those files when it is built." The command list and the paragraph "Most GIF demos animate…" stay until Task 14: the commands still write GIFs.

- [ ] **Step 4: Tests pass; full gate; render the site and run both checks.** `check_site_figures.py` prints `Figure contract OK: 10 pages, 18 figures checked.` Confirm that `site/_output/assets/` holds no GIF, and open the landing page and one story page in a browser: the poster shows, nothing plays until play is pressed, and the theme switch swaps the poster.

- [ ] **Step 5: Commit and push** as `Show the six demos as clips drawn from replay data; retire the GIFs`. The body states that the 2D clip still shows the run at M = 0.15, which Task 11 changes.

---

### Task 11: The 2D demo's true mass, 0.15 to 0.10, and the box-margin guard

**Files:**
- Modify: `src/clocks/_demos/demo_2d.py`, `assets/replays/demo_2d.npz` (re-recorded), `site/index.qmd`, `site/story/into-the-plane.qmd`, `docs/audits/2026-11-17/README.md`
- Test: `tests/test_replays.py`, `tests/test_site_figure_style.py`

**Interfaces:**
- Produces, in `tests/test_replays.py`: `BOX_MARGIN = 0.05`; `_box_margins(true_params: NDArray, prior_ranges: Sequence[Sequence[float]], dimensions: int) -> NDArray[np.float64]`, per parameter the smaller of its distances to the two edges of its prior range, divided by the range's width. With as many ranges as parameters they pair off in order; with two ranges (model comparison) the first serves every position and the second every mass, the truth being laid out positions first.

- [ ] **Step 1: Write the failing tests.**

  In `tests/test_replays.py`:

  - `test_box_margins` on literals: `_box_margins([0.0, 0.15], [[-8, 8], [0.005, 0.15]], 1)` is `[0.5, 0.0]`; `_box_margins([-7.2, 0.01225], ...)` is `[0.05, 0.05]` (`rtol=1e-9`); `_box_margins([-2.0, 3.0, 0.045, 0.03], [[-8, 8], [0.005, 0.15]], 1)` is `[0.375, 0.3125, 0.2758620689655173, 0.17241379310344826]` (`rtol=1e-12`). Only this catches the margin measured from one edge only, or the model-comparison truth paired with the wrong range.
  - `test_truth_sits_inside_its_prior_box` (IDs `scenario-<demo>` and `archive-<demo>` for the seven demos): from `_scenario(name)` or from the committed archive's `true_params` and `meta["prior_ranges"]`, every margin is at least `BOX_MARGIN`; the failure message names the parameter and its margin. Today `scenario-demo_2d` and `archive-demo_2d` fail with a margin of 0 for M. Only this catches a truth on or near its prior's edge, the audit's finding C6; a truth 6% inside passes, as the design says.

  In `tests/test_site_figure_style.py`:

  - `test_the_plane_cell_uses_the_demos_truth`: the plot cell of `story/into-the-plane.qmd` contains `f"masses=np.array([{demo_2d.TRUE_M:.2f}])"`, `f"positions=np.array([[{demo_2d.TRUE_X}, {demo_2d.TRUE_Y}]])"`, `mass_range=(0.005, 0.15)`, `n_observations=25, seed=42` and `n_particles=500`. It passes before the change (0.15 in both places) and after it only if both change. Only this catches the page's cell left at the old truth.
  - `test_the_plane_cells_knot_is_offset_by_about_its_width`: running that configuration through `simulate` and `infer` in the test (the cell's literals, with `demo_2d.TRUE_M`), the distance of the posterior mean from the true position, divided by the mean of the two position standard deviations, lies between 0.5 and 1.5 (0.94 at 0.10, Q6); and the page, whitespace-normalised, contains "to a small knot beside the true position, offset from it by about its own width" and not "tight knot at the truth". At 0.15 the ratio is 1.08 and the old sentence is still on the page, so this fails on the sentence until Step 3.

- [ ] **Step 2: Run** `uv run pytest tests/test_replays.py tests/test_site_figure_style.py -q -m ''`: the two `demo_2d` box cases and the sentence assertion fail.

- [ ] **Step 3: Implement, in this order, running the named tests after each.**
  1. `TRUE_M = 0.10` in `src/clocks/_demos/demo_2d.py`. Now `scenario-demo_2d` passes, and `test_archive_matches_its_scenario[demo_2d]`, `test_gate_inputs[demo_2d]` and `test_gate_summaries[demo_2d]` fail (measured for the reverse change, Q7), with `archive-demo_2d`.
  2. `uv run python scripts/record_replays.py --out assets/replays --overwrite demo_2d`. On arm64 macOS the archive's last row is (1.41692, −1.22229, 0.10002) ± (0.11108, 0.11479, 0.00124) (Q6); on another machine it may differ in low-order bits, and the gate compares a fresh run on the same machine. `uv run pytest -q -m '' tests/test_replays.py tests/test_demo_records.py tests/test_shipped_ground_truths.py tests/test_record_replays_script.py` passes (237 tests in the planning run, Q6), `test_a_tilted_covariance_is_caught` among them: its precondition holds at row 3, |r| = 0.690. `test_description_numbers_are_the_archives` now fails for the two `demo_2d` cases.
  3. In both descriptions of the 2D clip the numbered sentence becomes "After 50 readings the run ends at x = 1.42 ± 0.11, y = -1.22 ± 0.11 and M = 0.100 ± 0.001, against a true x = 1.50, y = -1.00 and M = 0.100." The rest stands: the y estimate is 1.94 of its standard deviations from the truth.
  4. In `site/story/into-the-plane.qmd`, the cell's `masses=np.array([0.15])` becomes `masses=np.array([0.10])`, and "The cloud has collapsed from a uniform smear over a 16×16 box to a tight knot at the truth — three parameters recovered from nothing but tick rates." becomes "The cloud has collapsed from a uniform smear over a 16×16 box to a small knot beside the true position, offset from it by about its own width — three parameters recovered from nothing but tick rates."
  5. In `docs/audits/2026-11-17/README.md`, the status table gains, above the "Open" row, `| C6 (2D truth on the prior's bound) | Fixed in the replay-clips pull request: the truth is 0.10, and a test keeps every shipped truth 5% inside its prior box |` and `| V1, V5 (animations; pinned media bytes) | Addressed in the replay track (#27, #28 and the replay-clips pull request): replay data is committed and the site draws its clips from it |`; C6, V1 and V5 leave the "Open" row. The heading's date and the audit's body are not edited; a line under the table says "Rows for C6, C9, V1 and V5 were added on YYYY-MM-DD."

- [ ] **Step 4: Tests pass; full gate; render the site and run both checks.** The partial rule does not apply (the render is full); confirm the 2D clip's final header reads M 0.100 ± 0.001 on the landing page, and that the cell's figure on "Into the Plane" shows the knot beside the ring. `git diff --stat main -- assets/replays` lists `demo_2d.npz` only.

- [ ] **Step 5: Commit and push** as `Lower the 2D demo's true mass to 0.10; keep every shipped truth inside its prior box`. The body gives the printed final estimate and says the archive was re-recorded on this machine.

---

### Task 12: "Run exactly once" becomes a statement of protocol

**Files:**
- Modify: `site/reproduce/reproducibility.qmd`, `site/reproduce/getting-started.qmd`, `site/method/the-particle-filter.qmd`, `README.md`, `docs/someday-maybe.md`, `docs/audits/2026-11-17/README.md`
- Test: `tests/test_generated_assets.py`

- [ ] **Step 1: Check the three citations, then write the failing tests.** Read, in the repository: `docs/superpowers/specs/2026-07-03-clone-freeze-diagnosis.md` (it says the configuration passed "only 7/12 on holdout seeds 100–111" and that "Seeds 100–111 are burned"); `docs/superpowers/specs/2026-07-02-annealed-jitter-design.md` ("Certification (seeds 200-211, run exactly once for tau=15, floor=0.02)"); `docs/superpowers/specs/2026-07-19-3d-echolocation-design.md` ("Certification (seeds 300-311, run once, 2026-07-19)"). If any of the three does not say what is quoted here, stop and report: the disclosure sentence is then left out, as the design requires.

  - `test_run_once_is_stated_as_protocol` (IDs the five files): the file, whitespace-normalised and lower-cased, does not contain "exactly once" and does contain "by protocol". Only this catches the attestation coming back in any of the five places.
  - `test_the_protocol_sentences`: the Reproducibility page contains "By protocol each reserved block is for one run: block 400–411 was run under the frozen settings, using the scenario-default cell via `--seed-block 400`, and protected blocks reject control overrides. The repository records that run; it does not enforce that there was only one."; Getting Started contains "By protocol the 400–411 block is reserved for one run, made under those frozen settings;"; the README contains "which by protocol is for one run and was run under those frozen settings."; the particle-filter page contains "and the reserved 400–411 block, which by protocol is for one run, was run under those frozen settings."
  - `test_the_earlier_seed_blocks_are_disclosed_with_their_records`: the Reproducibility page contains "Earlier implementations used three earlier blocks, as their design records state: seeds 100–111 as a holdout", and links, under `https://github.com/jbwhit/clocks/blob/main/`, the three files above; each file exists in the repository and contains its phrase: "holdout seeds 100–111", "seeds 200-211, run exactly once" and "seeds 300-311, run once". Only this catches a disclosure whose citation has gone.

- [ ] **Step 2: Run** `uv run pytest tests/test_generated_assets.py -q`; the three tests fail.

- [ ] **Step 3: Implement.**
  - Reproducibility: "The reserved 400–411 block was then run exactly once for the corrected implementation, using the scenario-default cell via `--seed-block 400`; protected blocks reject control overrides." becomes "By protocol each reserved block is for one run: block 400–411 was run under the frozen settings, using the scenario-default cell via `--seed-block 400`, and protected blocks reject control overrides. The repository records that run; it does not enforce that there was only one. Earlier implementations used three earlier blocks, as their design records state: seeds 100–111 as a holdout ([clone-freeze diagnosis](https://github.com/jbwhit/clocks/blob/main/docs/superpowers/specs/2026-07-03-clone-freeze-diagnosis.md)), and seeds 200–211 and 300–311 as certification blocks ([annealed-jitter design](https://github.com/jbwhit/clocks/blob/main/docs/superpowers/specs/2026-07-02-annealed-jitter-design.md), [3D echolocation design](https://github.com/jbwhit/clocks/blob/main/docs/superpowers/specs/2026-07-19-3d-echolocation-design.md))."
  - Getting Started: "The 400–411 block was then run exactly once; its tracked results are" becomes "By protocol the 400–411 block is reserved for one run, made under those frozen settings; its tracked results are".
  - README: "before the reserved certification block, which was then run exactly once." becomes "before the reserved certification block, which by protocol is for one run and was run under those frozen settings."
  - The particle-filter page and `docs/someday-maybe.md`: "and the reserved 400–411 block has now been run exactly once." becomes "and the reserved 400–411 block, which by protocol is for one run, was run under those frozen settings."
  - The audit's status table: `| C9 ("run exactly once") | Reworded as a statement of protocol in the replay-clips pull request, with the earlier seed blocks disclosed and cited; a test pins the wording |`, C9 leaves the "Open" row, and T2's row ends "…are literals). The protocol wording is pinned since the replay-clips pull request."

- [ ] **Step 4: Tests pass; full gate; render the site and run both checks; commit and push** as `State the one-run rule as protocol and disclose the earlier seed blocks`.

---

### Task 13: Contact sheets and the checks by eye

**Files:**
- Modify: `docs/plans/2026-11-19-replay-and-animation/contact_sheets.py`; the six `clip-<demo>.jpg` (redrawn); any clip module, caption or description that Steps 2 to 4 have to fix
- Delete: `docs/plans/2026-11-19-replay-and-animation/clip_probe.py`

- [ ] **Step 1: Draw the sheets with the real builders.** `contact_sheets.py` gains the six clips: for each demo and theme, `build_clip` on the committed archive inside `style(theme)`, shown at rows 0, 1, 8 and the last (with the shot's azimuth), each written with `save_png` and composed as Task 1's sheets were: full width above, 780 pixels below. It still draws `density.jpg` and `range-study.jpg`, imports only through `clocks.viz`, `clocks._clips` and `clocks._style`, and holds no absolute path. `git rm` the probe script: the builders now carry its layouts.

- [ ] **Step 2: Look at each sheet** and report each item of Task 1, Step 2 as seen or not seen, on the real clips. A failed item is fixed in the clip module, the clip tests are re-run, and the sheet is redrawn; say what moved.

- [ ] **Step 3: Read each description and caption against its clip**: what the prior looks like, the early change, the ending, and each word of layout. In particular, the two-mass 2D description's clause that each cloud "is split between two arrangements of the masses": if the frames between rows 1 and 80 do not show it, delete the clause. A changed description must still pass `test_description_numbers_are_the_archives`.

- [ ] **Step 4: Check the pages in browsers**, on the site as rendered at this commit, served from `site/_output`.
  - Headless Chrome, against a server that logs requests, as the design's M12: on each of the six pages, both posters are requested and no `.mp4` is requested before play, with JavaScript on and off; each visible video's box is 750 by its declared height before any media loads; the theme toggle swaps which video is visible.
  - Phone width, pull request #21's procedure: each of the six pages in a 390-pixel-wide `<iframe>`, once per theme: `document.documentElement.scrollWidth` is 390, and each visible video's width is above 0 and at most its container's; the hidden one reports 0.
  - Chrome, Firefox and Safari by hand: the poster and controls with JavaScript off; play, pause and seek from the keyboard; the theme switch. An agent cannot do the Firefox and Safari checks; report them as not done, and the coordinator asks the owner before the pull request is marked ready. Safari needs byte-range responses, which GitHub Pages is believed to serve (design, "Risks"); it can only be confirmed on the deployed site.

- [ ] **Step 5: Check the sheets' metadata** as in Task 1, Step 6; tests pass; full gate; render the site and run both checks.

- [ ] **Step 6: Commit and push** the script, the sheets and any fix as `Redraw the clip contact sheets from the real builders; record the checks by eye`. The body lists what Steps 2 to 4 found and what is left for a person.

**The split point.** After this task the site is correct and complete, every renderer added so far is called by the site's build, the GIFs are gone and the 2D truth is 0.10. The commands still write GIFs through `_animate.py`, and the README and Getting Started say so. If the pull request is split, this is the recommended place; the earliest possible one is after Task 10 (decision D3).

---

### Task 14: The demo commands: `--output-dir`, `--theme`, `--video`

**Files:**
- Modify: `src/clocks/_demos/_common.py` and the seven modules beside it; `scripts/check_wheel_entrypoints.py`; `README.md`; `site/reproduce/getting-started.qmd`; `site/reproduce/reproducibility.qmd`
- Test: `tests/test_cli.py`, `tests/test_density_demo.py`, `tests/test_generated_assets.py`

**Interfaces:**
- Consumes: `final_frame`, `render_clip` from `clocks._clips`; `find_encoder`, `EncoderNotFound`, `EncoderError` from `clocks._video`; `save_replay`, `Replay` from `clocks._replay`; `style`, `save_png`, `THEMES`, `Palette` from `clocks._style`; `trace_decimals`, `density_figure` from `clocks._figures`.
- Produces, in `src/clocks/_demos/_common.py`:

```python
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

  - `add_common_arguments` adds `--output-dir DIR` (a `Path`, default `Path("output")`), `--observations`, `--particles` (as today) and `--theme` (choices `light` and `dark`, default `light`). There is no `--output`. `add_video_argument` adds `--video` (`store_true`).
  - `run_command` parses `argv`; sets `args.encoder` to `None`, or, when the parser has `--video` and it was given, to `find_encoder()`, calling `parser.error(str(error))` on `EncoderNotFound` (exit status 2, before `run` is called); calls `run(args)`; turns an `EncoderError` from it into `parser.exit(1, f"{parser.prog}: error: {error}\n")`; returns 0.
  - `ascii_name` maps `₁` and `₂` to `1` and `2`, `μ` to `mu` and `σ` to `sigma`, so a command's output is ASCII apart from `±`, as it is today.
  - `print_truth(replay)` prints `Truth: ` followed by `f"{ascii_name(name)}={value:g}"` for each parameter, joined by `, `, then `f"True rates: {replay.arrays['true_rates']}"`.
  - `print_estimate(replay)`, for a filter replay, prints `f"Final estimate after {T} observations:"`, then per parameter `f"  {ascii_name(name)} = {mean:.{d}f} ± {sd:.{d}f}  (true: {truth:g})"` with the last row's values and `d = trace_decimals(prior range) + 1`, then `f"  ESS = {ess:.0f} / {n_particles}"`, all from the replay.
  - `write_outputs` creates `directory` if missing and writes, in this order: `<directory>/<demo>.npz` with `save_replay` (an existing file is replaced whole); `<directory>/<demo>.png`, the figure `figure(replay, colors)` returns inside `with style(theme) as colors:`, with `save_png` (an existing file is overwritten in place), the figure closed in a `finally`; and, when `encoder` is given, `<directory>/<demo>.mp4` with `render_clip(replay, path, theme=theme, encoder=encoder)` (replaced whole, or left as it was on failure). It prints `f"Saved: {path}"` after each file and returns the paths. `<demo>` is `replay.meta["demo"]`.
- Each of the six clip demos: `build_parser()` calls `add_common_arguments` and `add_video_argument`; `_run_demo(args)` is `replay = record(n_observations=args.observations, n_particles=args.particles)`, `print_truth(replay)`, `print_estimate(replay)`, `write_outputs(replay, args.output_dir, args.theme, final_frame, args.encoder)`; `main(argv)` is `return run_command(build_parser(), _run_demo, argv)`. `OUTPUT_PATH` and every import only the old path used go. The model-comparison command prints, in place of `print_estimate`: `f"True model: K={true_k}"` first; for k = 20, 40, … up to T, `f"After {k} observations:"` and one `f"  K={K}: posterior={p:.4f}"` per candidate; then `f"MAP model: K={best}"`, `f"Estimate: {final_mean}"`, `f"Std:      {final_std}"` and `f"ESS:      {ess:.0f} / {n_particles}"` for the candidate with the largest final probability, all from the replay. It records once.
- `demo_density`: the same, without `add_video_argument`, with `density_figure` as the figure; its printing is as pull request 2 left it.
- `scripts/check_wheel_entrypoints.py` gains one run after the `--help` loop, in a new empty directory as the working directory and with an environment whose `PATH` is the temporary environment's scripts directory alone: `demo-1d --output-dir out --observations 2 --particles 16` must exit 0 and leave `out/demo_1d.npz` and `out/demo_1d.png`; the same command with `--video` must exit non-zero with `imageio-ffmpeg` in its standard error and leave no `out/demo_1d.mp4`.

- [ ] **Step 1: Write the failing tests.** In `tests/test_cli.py`, `DEMO_CASES` holds module and executable names only; `CLIP_DEMOS` is the six with a clip. Each test below runs in `tmp_path` and passes `--output-dir`.

  - `test_demo_parser_defaults_and_overrides` (each demo): the defaults are `output_dir == Path("output")`, `theme == "light"`, `observations > 1`, `particles > 16`, and `video is False` for a clip demo; `--output-dir x --observations 1 --particles 16 --theme dark` parses to those values. Only this catches a changed default directory or theme.
  - `test_output_is_no_longer_an_option` (each demo): `main(["--output", "x.png"])` raises `SystemExit` with code 2. Only this catches `--output` kept as an alias the documentation says is gone.
  - `test_demo_main_writes_its_archive_and_png` (each demo): `main(["--output-dir", str(tmp_path / "new" / "out"), "--observations", "1", "--particles", "16"]) == 0`; the directory holds exactly `<demo>.npz` and `<demo>.png`; the archive loads with `meta["demo"] == <demo>`, `n_observations == 1` and `n_particles == 16`; the PNG's size is 1500 by twice the demo's declared height (1080 for `demo_density`). Only this catches a command that writes under another name, draws another run than it archived, or does not create the directory.
  - `test_theme_dark_draws_the_dark_png` (each demo): with `--theme dark`, the PNG's first pixel is `palette("dark").background`; without it, the light one. Only this catches `--theme` parsed and ignored.
  - `test_png_is_the_final_frame` (each clip demo): with `clocks._demos._common.save_png` wrapped by a spy and `final_frame` wrapped by a spy on the module that calls it, `save_png` is called once, with the figure `final_frame` returned (`is`) and the path `<dir>/<demo>.png`, and `final_frame` once, with the replay that was archived. Only this catches a PNG of a different figure.
  - `test_outputs_replace_existing_files`: for `demo_1d`, with `demo_1d.npz` and `demo_1d.png` holding `b"old"` beforehand, the run leaves a loadable archive and a PNG. Only this catches a command that refuses or appends to an existing output.
  - `test_video_writes_the_clip` (each clip demo): `--video --observations 2 --particles 16` leaves exactly the three files, and `probe` of the clip reports 92 frames of 1500 by twice the declared height. Only this catches `--video` writing nothing, or a clip of another run.
  - `test_density_has_no_video_option`: `main(["--video"])` for `demo_density` raises `SystemExit` with code 2.
  - `test_video_without_an_encoder_stops_before_the_run` (each clip demo): with `clocks._demos._common.find_encoder` raising `EncoderNotFound` (the library's own message) and the module's `record` replaced by `pytest.fail`, `main(["--output-dir", str(tmp_path / "out"), "--video"])` raises `SystemExit` with code 2; standard error contains `imageio-ffmpeg` and `ffmpeg on PATH`; `tmp_path / "out"` does not exist. Only this catches a minute of inference before the command says it cannot make the clip (Review Focus 3).
  - `test_a_failed_clip_keeps_the_archive_and_png`: for `demo_1d`, with `find_encoder` returning `"unused"` and `clocks._demos._common.render_clip` raising `EncoderError("the encoder failed")`, `main` raises `SystemExit` with code 1; standard error ends `error: the encoder failed`; the directory holds exactly the archive and the PNG. Only this catches a traceback for a failed encode, or outputs removed because the clip failed.
  - `test_command_prints_the_recorded_run` (the five clip demos with a filter archive): the module's `record` replaced by a stub that notes its keywords and returns `small(name)` with `true_rates[0]` set to 0.5 and 0.25 times the first prior width added to `mean[-1, 0]`; `main` with `--observations 7 --particles 23` calls it once with those two values; standard output holds the line `f"True rates: {edited}"`, the `Truth:` line built in the test from the stub's `true_params`, and after `Final estimate after 9 observations:` the lines built in the test from the stub's last row with `trace_decimals(...) + 1` decimals and the stub's `ess` and `n_particles`. Only this catches numbers printed from constants or from the arguments.
  - `test_model_comparison_command_prints_from_the_replay_and_records_once`: the same pattern on `demo_model_comparison` with `record` wrapped by a counting spy: one call; the output holds `True model: K=2`, the `MAP model:` line for the candidate of largest final probability, and `Estimate:` and `Std:` lines equal to `str` of that candidate's `final_mean_k<K>` and `final_std_k<K>`; and the module has no attribute `infer`. Only this catches the command running the comparison a second time for its printout.
  - `test_ascii_name`: `x₁` → `x1`, `M₂` → `M2`, `μ` → `mu`, `σ` → `sigma`, `y` → `y`.
  - `test_demo_help_exits_without_running`, `test_project_scripts_point_directly_to_packaged_demo_modules`, `test_demo_selects_agg_before_importing_pyplot` and `test_source_demo_is_only_a_thin_wrapper` are kept, with the changed `DEMO_CASES`.

  In `tests/test_density_demo.py`: every `main([...])` passes `--output-dir <dir>` and reads `<dir>/demo_density.png`. `test_density_command_writes_png_bytes_at_the_path_given` is deleted (the command no longer takes a path), and `test_density_command_writes_only_its_png` becomes `test_density_command_writes_exactly_its_archive_and_png`: the directory holds exactly `demo_density.npz` and `demo_density.png`, and the archive's `meta` has the requested counts. Its spies move from `demo_density.save_png` to `clocks._demos._common.save_png`.

  In `tests/test_generated_assets.py`:

  - `test_command_lists_name_the_files_the_commands_write` (IDs `README.md`, `site/reproduce/getting-started.qmd`): for each of the seven commands, the file has a line starting `uv run <executable> ` that contains `output/<demo>.npz` and `output/<demo>.png`; a line starting `uv run demo-2d --video` contains `output/demo_2d.mp4`; and the file does not contain `.gif`. Only this catches a command list that still promises GIFs.
  - `test_commands_provenance_sentence`: the Reproducibility page contains "Each `uv run demo-*` command records the same configured scenario again, prints its estimates, and writes an archive and a PNG of the final frame (for `demo-density`, the density figure); `--video` also writes the clip." and no longer contains "`uv run demo-density` records the same configured scenario and draws its result."

- [ ] **Step 2: Stage the names, then run.** Add the new functions to `_common.py` (`run_command` parsing and calling `run` only; `write_outputs` returning `[]`; the printers printing nothing; `ascii_name` returning its argument), keeping the old `add_common_arguments` signature working beside the new one for this step only. Run `uv run pytest tests/test_cli.py tests/test_density_demo.py tests/test_generated_assets.py -q`: the new tests fail on their assertions (GIFs written, `--output` accepted, nothing printed from the replay).

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

  with "Six animated demos and one static figure:" replaced by "Seven demos:", and the paragraph "Most GIF demos animate…" by: "Each command records its run, prints the truth and its final estimate, and writes a replay archive and a PNG of the final frame (for `demo-density`, the density figure). `--video` also writes the animation as an MP4; it needs an FFmpeg encoder, which `uv sync` provides in a checkout. `--theme dark` draws the dark version and `--output-dir` chooses the directory. All seven, with commentary: [jbwhit.github.io/clocks](https://jbwhit.github.io/clocks/). The echolocation range study behind the site's final page: `scripts/scan_echolocation_range.py`." Getting Started gets the same block under "Seven demos. Each writes a replay archive and a PNG; `--video` adds an MP4:". On the Reproducibility page the sentence about `demo-density` becomes the one the test pins.

- [ ] **Step 4: Tests pass; full gate.** Then `uv build` and `uv run python scripts/check_wheel_entrypoints.py dist/*.whl python3`, which must end with the two new `ok:` lines (`dist/` is git-ignored). The script has no unit test: it is a check, run here and by CI's "Smoke-test installed entry points" step. Render the site and run both checks (two pages changed).

- [ ] **Step 5: Commit and push** as `Demo commands record, print from the replay and write an archive and a PNG; --video adds the clip`. The body says that `--output` is gone and what each command now writes.

---

### Task 15: The public names, the old modules, the version and the upgrade note

**Files:**
- Delete: `src/clocks/_animate.py`, `src/clocks/_panels.py`, `src/clocks/_panels3d.py`, `tests/test_viz.py`
- Modify: `src/clocks/viz.py`, `pyproject.toml`, `uv.lock`, `README.md`, `site/reproduce/architecture.qmd`, `docs/someday-maybe.md`
- Test: `tests/test_api.py`, `tests/test_generated_assets.py`

- [ ] **Step 1: Confirm nothing else uses what is being deleted.** `grep -rn "_animate\|_panels\|animate_\|create_.*dashboard\|plot_clock\|plot_particle\|plot_mass\|plot_scene\|plot_centered" src scripts site tests README.md` must find only `src/clocks/viz.py`, the three modules themselves and `tests/test_viz.py` (planning found exactly these once the demos stop importing the old names in Task 14, plus the README and Architecture lines this task rewrites: Q12). `grep -n "^class \|^def test" tests/test_viz.py` lists eighteen classes, each of which exercises only removed names; if one tests something that survives, move it to `tests/test_clips.py` and say so.

- [ ] **Step 2: Write the failing tests.**

  In `tests/test_api.py`:

  - `test_viz_exports_exactly_the_documented_names`: `set(clocks.viz.__all__) == {"THEMES", "Palette", "density_figure", "load_replay", "palette", "range_study_figure", "render_clip", "show", "style"}`, and each is an attribute. Only this catches a name added or dropped without the upgrade note changing.
  - `test_the_removed_names_are_gone`: with `ANIMATE` the six names `animate_echolocation`, `animate_inference`, `animate_inference_2d`, `animate_inference_multi_1d`, `animate_inference_multi_2d`, `animate_model_comparison`; `PLOT` the eleven `plot_centered_rates`, `plot_clock_rates`, `plot_clock_rates_2d`, `plot_clock_setup`, `plot_clock_setup_2d`, `plot_mass_histogram`, `plot_particle_cloud`, `plot_particle_cloud_2d`, `plot_particle_cloud_multi_1d`, `plot_particle_cloud_multi_2d`, `plot_scene_3d`; and `DASHBOARD` the two `create_echolocation_dashboard`, `create_inference_dashboard`: none is an attribute of `clocks.viz`, and no module `clocks._animate`, `clocks._panels` or `clocks._panels3d` can be imported. Only this catches an old name left importable.
  - `test_the_version_is_0_2_0_in_the_project_and_the_lock`: with `tomllib`, `pyproject.toml`'s `project.version` is `"0.2.0"`, and the entry of `uv.lock`'s `package` list named `clocks` has `version == "0.2.0"`. Only this catches a version bump without `uv lock`, which fails CI's `uv sync --locked` (Q11).

  In `tests/test_generated_assets.py`:

  - `test_the_readme_has_the_upgrade_note`: the README contains the heading `## Upgrading from 0.1` and, whitespace-normalised, each of: "Removed from `clocks.viz`: the six `animate_*` functions, the eleven `plot_*` functions and the two `create_*_dashboard` functions."; "Added: `load_replay`, `render_clip`, `density_figure`, `range_study_figure`."; "`--output PATH` becomes `--output-dir DIR`"; "Each demo writes a replay archive and a PNG where it wrote a GIF; an MP4 is opt-in, with `--video`." The counts six, eleven and two are those of the test above.
  - `test_the_architecture_page_names_only_modules_that_exist`: every `_name` in the page's Mermaid block that is written as a node label (the text inside `["..."]`, split on `, `, a trailing `*` or `/` dropped, `_clip_` standing for the `_clip_*.py` files) names a module or directory under `src/clocks/`; the page contains none of `_animate`, `_panels`. Only this catches the diagram drifting from the package again.
  - `test_the_readme_structure_names_only_files_that_exist`: every entry of the README's `src/clocks/` listing (a line of the code block under "Project structure" that begins with four spaces and a file or directory name) exists under `src/clocks/`; the listing contains none of `_animate`, `_panels`.
  - `test_someday_maybe_no_longer_says_static_gifs`: `docs/someday-maybe.md` does not contain "static GIFs" and contains "pre-rendered video drawn from committed replay data".

- [ ] **Step 3: Run** `uv run pytest tests/test_api.py tests/test_generated_assets.py -q`; the new tests fail.

- [ ] **Step 4: Implement.**
  - `git rm` the three modules and `tests/test_viz.py`. `clocks.viz` imports and exports exactly the nine names; its docstring reads "Figures, clips and the shared style (public facade). Implementations live in `clocks._figures` and `clocks._echo_study` (figures drawn from committed data), `clocks._clips` (clips drawn from replays), `clocks._replay` (the archive loader) and `clocks._style`."
  - `pyproject.toml`: `version = "0.2.0"`; then `uv lock`, and confirm `git diff uv.lock` changes the `clocks` version line and nothing else.
  - `README.md`, a section before "Run tests":

```markdown
## Upgrading from 0.1

Version 0.2.0 changes what the demos write and removes the old plotting names.

- Removed from `clocks.viz`: the six `animate_*` functions, the eleven `plot_*`
  functions and the two `create_*_dashboard` functions. Added: `load_replay`,
  `render_clip`, `density_figure`, `range_study_figure`.
- `--output PATH` becomes `--output-dir DIR`; the file names are the demo's.
- Each demo writes a replay archive and a PNG where it wrote a GIF; an MP4 is
  opt-in, with `--video`.
```

    and in "Project structure" the `viz.py` line becomes `viz.py         Figure, clip and style facade (_figures.py, _clips.py, _replay.py, _style.py)`, the `_panels3d.py` line goes, and these are added in the listing's order: `_replay.py     Replay archives: record, validate, load, compare`; `_figures.py    Static figures drawn from a replay, and the trace helpers`; `_clips.py      Clips drawn from replays (_clip_*.py hold the six builders)`; `_video.py      Raw frames to H.264 and back, through FFmpeg`; `_site_media.py The site's pre-render step: clips and posters into site/media/`.
  - `site/reproduce/architecture.qmd`: in the Mermaid block the `vis` subgraph's nodes become `viz`, `figures["_figures"]`, `clips["_clips, _clip_*"]`, `video["_video"]`, `sitemedia["_site_media"]` and `figstyle["_style, _fonts/"]`; `replay["_replay"]` joins the `scenarios` subgraph; the eight edges from `viz` and `animate` are replaced by `viz --> figures`, `viz --> clips`, `viz --> echo`, `viz --> replay`, `viz --> figstyle`, `figures --> replay`, `figures --> figstyle`, `clips --> figures`, `clips --> video`, `clips --> replay`, `clips --> figstyle`, `sitemedia --> clips`, `sitemedia --> video`, `sitemedia --> replay`, `replay --> api`, `echo --> figstyle`; and `demos --> viz` becomes `demos --> clips`, `demos --> figures` and `demos --> replay`. Before writing each new edge, confirm it with `grep -n "^from clocks\|^import clocks" <the module>`, and leave out any the imports do not show. In "In words", "`viz` is the public facade over private animation and panel modules and the shared figure style" becomes "`viz` is the public facade over the private figure and clip modules (`_figures`, `_clips`), the replay loader (`_replay`) and the shared figure style".
  - `docs/someday-maybe.md`: "(The website ships static GIFs by deliberate scope choice; see docs/superpowers/specs/2026-06-09-deep-clean-and-website-design.md.)" becomes "(The website ships pre-rendered video drawn from committed replay data, with native controls and no script; see docs/plans/2026-11-19-replay-and-animation-design.md.)"

- [ ] **Step 5: Tests pass; full gate; render the site and run both checks; build the wheel and run the entry-point check again** (the package's contents changed).

- [ ] **Step 6: Commit and push** as `Remove the GIF animators and panel functions; version 0.2.0 with an upgrade note`.

---

### Task 16: Mutation checks

No files change. This task proves the new tests can fail. **The coordinator runs it, with a mutation harness in a separate worktree** that no reviewer or implementer is using. The harness sets `PYTHONDONTWRITEBYTECODE=1`, deletes every `__pycache__` before each mutant, confirms `git status --porcelain` is empty before and after each, applies each mutant and restores it in a `try`/`finally`, and runs **the whole suite, default and slow tests together (`uv run pytest -q -m ''`), for every mutant**, page, data and prose mutants included, in the foreground with a 600000 ms timeout, waiting with the bracketed `pgrep` loop of Global Constraints if a run outlasts it. The baseline on the clean tree is recorded first.

Every mutant breaks library code, a script, a page or a data file, never a test. An ID in square brackets is a parametrized case; "each clip" means the six clips' cases, "each theme" both. Where a mutant names a function shared by several clips, it is applied in the shared function. The first column is the task that owns the code, so the table splits at Task 13 if the pull request does.

| Task | Mutation | Tests that must fail, and no others |
|---|---|---|
| 2 | `dwell_frames`: the counts for observations 4 to 8 and 9 to 20 exchanged | `test_dwell_frames[50]`, `[80]`, `[9]`, `[21]`; `test_echo_shots_turn_the_camera_and_hold` |
| 2 | `dwell_frames(0)` returns `[24]` | `test_dwell_frames_needs_an_observation` |
| 2 | `find_encoder`: `shutil.which` tried before the package | `test_find_encoder_order[package_first]` |
| 2 | `find_encoder`: the package's `RuntimeError` not caught | `test_find_encoder_order[package_without_a_binary]` |
| 2 | `_video.py`: `import imageio_ffmpeg` moved to module level inside a `try` | `test_video_module_imports_only_the_standard_library` |
| 2 | `encode`: `"-crf", "20"` replaced by `"-crf", "23"` | `test_the_encoder_is_given_the_designs_settings` |
| 2 | `encode`: `"-movflags", "+faststart"` removed | `test_the_encoder_is_given_the_designs_settings`; `test_encode_then_probe`; `test_render_clip_encodes_every_shot` (each clip); `test_a_full_render_writes_each_clip_and_poster` |
| 2 | `encode`: the temporary file moved into place whatever the encoder's exit status | `test_a_failed_encode_leaves_the_old_clip_and_nothing_else` |
| 2 | `encode`: frames piped straight to `path`, no temporary file | `test_a_failed_encode_leaves_the_old_clip_and_nothing_else`; `test_an_error_while_drawing_leaves_no_clip`; `test_encode_refuses_a_frame_of_the_wrong_length` |
| 2 | `encode`: the encoder looked up after the first frame is taken | `test_encode_looks_for_the_encoder_first` |
| 2 | `encode`: the even-size check removed | `test_encode_refuses_odd_sizes[width]`, `[height]` |
| 2 | `parse_stream_line`: the rate read from the number before ` tbr` | `test_parse_stream_line[fractional_rate]` |
| 2 | `box_order`: the walk stops at a size of 1 | `test_box_order` |
| 2 | `probe`: `moov_first` always true | `test_probe_sees_moov_after_mdat`; `test_a_video_without_faststart` |
| 2 | `probe`: a file with no `Video:` line gives `StreamInfo("", "", 0, 0, 0.0, 0, False)` | `test_probe_refuses_a_file_that_is_not_video`; `test_a_file_that_is_not_a_video` |
| 2 | the shared save function of `_style.py`: `facecolor=fig.get_facecolor()` removed | `test_save_png_writes_the_authored_canvas[hostile]`; `test_frame_rgba_is_what_save_png_writes[hostile]` |
| 2 | `pyproject.toml`: `imageio-ffmpeg` moved to `project.dependencies` (and `uv lock`) | `test_the_encoder_is_a_development_dependency_only` |
| 3 | `new_figure`: `plt.figure(figsize=..., layout="none")` and no `set_layout_engine` | `test_the_layout_engine_stays_off` (each clip); `test_a_one_particle_run_draws` (each clip) |
| 3 | `status_text(1)` returns `"After 1 observations"` | `test_status_text`; `test_status_line` (each clip, each theme) |
| 3 | `draw_traces`: the mean line holds every row from the first frame | `test_traces_follow_the_archive` (each clip with traces, each theme) |
| 3 | `draw_traces`: its update calls `ax.relim()` and `ax.autoscale_view()` | `test_axes_never_move` (each clip with traces); `test_traces_follow_the_archive` (each clip with traces, each theme) |
| 3 | `ellipse_points`: the factor 2 replaced by 1 | `test_ellipse_points_lie_two_standard_deviations_out`; `test_ellipse_is_the_two_standard_deviation_contour` (each ellipse) |
| 3 | `ellipse_points`: the off-diagonal entries of `cov` set to 0 | the same two |
| 3 | `ellipse_points`: `np.linalg.cholesky` in place of the eigen-decomposition | `test_ellipse_points_of_no_spread`; `test_a_one_particle_run_draws` (each clip with an ellipse) |
| 3 | `zoom_window`: centred on the final mean | `test_zoom_window`; `test_inset_is_the_fixed_window_on_the_truth` (each inset) |
| 3 | `zoom_window`: the zero-width fallback removed | `test_zoom_window` |
| 3 | `zoom_label`: always the one-number form | `test_zoom_label` |
| 3 | `clock_fill`: the blend parameter not clipped | `test_clock_fill[light]`, `[dark]`; `test_2d_a_reading_above_one_gives_a_background_fill` |
| 3 | `build_2d`: `largest` taken from the last row of the running means | `test_2d_clock_fills_follow_the_running_mean[light]`, `[dark]` |
| 3 | `build_2d`: each row's dots (map and inset alike) given y then x | `test_dots_are_the_archive_row[demo_2d: map dots]` |
| 3 | `build_2d`: the dots always taken from row 0 | `test_dots_are_the_archive_row[demo_2d: map dots]`; `test_changing_a_middle_cloud_row_moves_only_that_rows_dots[demo_2d: map dots]`; `test_the_poster_is_the_last_row_of_its_archive` |
| 3 | `draw_dots`: `alpha=0` | `test_dots_are_the_archive_row` (each cloud) |
| 3 | `build_2d`: the clocks' `zorder` above the dots' | `test_dots_are_the_archive_row[demo_2d: map dots]` |
| 3 | `build_2d`: the truth mark placed at the final mean | `test_truth_marks_sit_on_the_truth[demo_2d: truth]` (each theme) |
| 3 | `draw_status`: the text drawn with `color="0.5"` | `test_library_drawing_code_has_no_unstyled_colour[src/clocks/_clip_parts.py]`; `test_status_line` (each clip, each theme); `test_colours_are_the_palette_or_an_ink_blend` (each clip, each theme): mid grey is 7.8 levels (light) and 12.0 (dark) from the nearest point of the background-to-ink segment, against the helper's tolerance of 1.5 (measured) |
| 3 | `build_2d`: the key text `stronger fill = slower clock` removed | `test_2d_labels` |
| 3 | `final_frame`: shows the first shot | `test_final_frame_shows_the_last_row` (each clip); `test_the_poster_is_the_last_row_of_its_archive` |
| 3 | `render_clip`: `style("light")` whatever `theme` says | `test_render_clip_draws_in_the_theme_asked_for[dark]` |
| 3 | `render_clip`: the figure closed only after a successful encode | `test_render_clip_closes_its_figure_when_the_encoder_fails` |
| 3 | `render_clip`: the clip built before the encoder is looked up | `test_render_clip_looks_for_the_encoder_before_drawing` |
| 3 | `build_clip`: an unknown demo falls back to `build_2d` | `test_build_clip_refuses_a_demo_without_a_clip` |
| 3 | `_clip_maps.py`: `from clocks.physics import clock_rates` added (unused) | `test_clip_modules_import_no_model_code` |
| 4 | track clips: the predicted bars' half-length taken from `predicted_std[0]` | `test_slowing_marks_follow_the_archive` (both clips, each theme) |
| 4 | track clips: the observed bars' half-length `noise_std`, without the root | the same four |
| 4 | track clips: the observed marks visible at row 0 | the same four |
| 4 | track clips: `1 - predicted_mean` replaced by `predicted_mean` | the same four; `test_slowing_limits_hold_every_row` does not fail (it reads `slowing_limits`, which is not changed) |
| 4 | `SIDE_OFFSET = 0.0` | `test_slowing_marks_stand_side_by_side` (both clips) |
| 4 | `slowing_limits`: rows k ≥ 1 only for the predictions | `test_slowing_limits_hold_every_row` |
| 4 | track clips: the strip's x-limits left to autoscale | `test_track_panels_share_the_x_scale` (both clips) |
| 4 | `build_multi_1d`: the second cloud's mass column 2 in place of 3 (cloud and inset alike) | `test_dots_are_the_archive_row[demo_multi_mass: cloud dots-2]` |
| 4 | `build_multi_1d`: the second ellipse from the block (0, 2) | `test_ellipse_is_the_two_standard_deviation_contour[demo_multi_mass: ellipse-2]` |
| 4 | `build_multi_1d`: the second cloud drawn with `marker="o"` | `test_two_masses_are_told_apart_without_colour` (each theme) |
| 5 | `build_multi_2d`: the six traces placed in parameter order, three to a column | `test_two_mass_map_has_one_trace_column_per_mass` |
| 5 | `build_multi_2d`: the labels `mass 1` and `mass 2` exchanged | `test_two_mass_map_labels_each_truth` (each theme) |
| 5 | `build_models`: line colours `ordered[K - 1]` | `test_model_colours_go_by_position_in_model_ks` (each theme) |
| 5 | `build_models`: `(true model)` appended where K is 2 | `test_model_labels_keep_their_places_and_name_the_truth` (each theme) |
| 5 | `build_models`: each label's y set to its line's last value | the same two |
| 5 | `build_models`: connectors not updated after row 0 | `test_model_connectors_join_head_to_label` |
| 5 | `build_models`: the lines' first point is row 1 | `test_model_lines_grow_from_the_prior` (each theme); `test_changing_a_posterior_row_moves_only_that_point`; `test_model_connectors_join_head_to_label` does not fail (heads are unchanged) |
| 5 | `build_models`: the candidate limit removed | `test_model_clip_refuses_a_fourth_candidate` |
| 6 | `azimuth`: `-75 + 40 * frame / (frames - 1)` | `test_azimuth` |
| 6 | `shots`: F taken as the whole frame count, hold included | `test_echo_shots_turn_the_camera_and_hold` |
| 6 | `build_echo`: the azimuth it is given ignored | `test_echo_camera_follows_the_shot` |
| 6 | `build_echo`: the scene's dots not cropped | `test_echo_close_view_holds_exactly_the_dots_inside_the_cube` |
| 6 | `in_view`: strict inequalities | `test_in_view_counts_the_faces` |
| 6 | `view_weight_text`: the prior's wording at every row | `test_view_weight_text` |
| 6 | `build_echo`: the fit panel's x and y exchanged | `test_echo_fit_panel` (each theme) |
| 6 | `build_echo`: the scene's limits recomputed from each row's dots | `test_echo_scene_is_the_fixed_cube`; `test_axes_never_move[demo_echolocation_3d]` |
| 6 | `build_echo`: the truth ring filled | `test_echo_truth_ring_and_drop_line` (each theme) |
| 6 | `build_echo`: pane colours left at Matplotlib's defaults | `test_colours_are_the_palette_or_an_ink_blend[demo_echolocation_3d]` (each theme) |
| 6 | `CLIP_HEIGHTS` gains `"demo_density": 540` | `test_every_clip_demo_has_a_builder`; `test_site_media_covers_the_clip_demos` |
| 7 | parser: a themed `<video>` in a replay figure treated as a plot wrapper | every clip test of `tests/test_check_site_figures.py` whose page holds a well-formed pair of videos (the two positive tests and the negative ones from `test_wrong_clip_height` on), which now report `expected a clip, found a plot wrapper`; list them from the run |
| 7 | the kind check removed | `test_a_plot_where_a_clip_is_listed`; `test_a_clip_where_a_plot_is_listed`; `test_a_themed_video_outside_a_replay_figure`; `test_a_clip_out_of_order` |
| 7 | each attribute or file check removed, one mutant per check | its negative fixture of Task 7 and no other: `autoplay` → `test_autoplay`; `loop` → `test_loop`; `controls` → `test_no_controls`; `playsinline` → `test_no_playsinline`; `preload` → `test_preload_auto`; `aria-label` → `test_missing_aria_label`, `test_blank_aria_label`; the description's existence → `test_missing_description`; its emptiness → `test_empty_description`; its uniqueness → `test_two_elements_share_the_description_id`; equality between themes → `test_description_differs_between_themes`; the caption count → `test_no_caption`; the empty caption → `test_empty_clip_caption`; width → `test_wrong_clip_width`; height → `test_wrong_clip_height`; the file names → `test_wrong_clip_name`, `test_wrong_theme_file`; the same-file check → `test_wrong_theme_file`; video content → `test_identical_light_and_dark_videos`; poster content → `test_identical_light_and_dark_posters`; the poster's size → `test_poster_of_the_wrong_size`; the PNG signature → `test_poster_that_is_not_a_png`; one replay figure → `test_videos_in_two_replay_figures` |
| 7 | the frame-count comparison removed | `test_a_truncated_video`; `test_a_video_of_the_wrong_frame_count` |
| 7 | the codec comparison removed | `test_a_video_in_another_codec` |
| 7 | the pixel-format comparison removed | `test_a_video_in_another_pixel_format` |
| 7 | the stylesheet check removed | `test_the_responsive_rule_is_required` |
| 7 | the stylesheet check made unconditional | `test_plot_pages_need_no_stylesheet`, and every plot-only test of the file that asserts an exact list of failures; list them from the run |
| 7 | a missing encoder ignored | `test_no_encoder` |
| 8 | `render_site_media`: the archive's time left out of the staleness rule | `test_a_partial_render_rebuilds_what_is_stale[archive_newer]`, `[tie]` |
| 8 | the sources' time left out | `test_a_partial_render_rebuilds_what_is_stale[source_newer]` |
| 8 | "not strictly newer" relaxed to "older" | `test_a_partial_render_rebuilds_what_is_stale[tie]` |
| 8 | only the clip's existence checked | `test_a_partial_render_rebuilds_what_is_stale[poster_missing]` |
| 8 | the newer of the two files compared | `test_a_partial_render_rebuilds_what_is_stale[clip_older]` |
| 8 | `__pycache__` not skipped | `test_a_cached_bytecode_file_does_not_make_media_stale` |
| 8 | the cleanup run on a partial render too | `test_a_partial_render_rebuilds_what_is_stale[stray_file_kept]` |
| 8 | the cleanup removes every other file, whatever its suffix | `test_a_full_render_removes_other_media_and_nothing_else` |
| 8 | the cleanup left out | the same |
| 8 | the directory created before the encoder is looked up | `test_no_encoder_stops_before_anything_is_touched` |
| 8 | the poster saved straight to its final name | `test_a_failed_poster_leaves_the_pair_stale` |
| 8 | the poster drawn in the light theme for both | `test_a_full_render_writes_each_clip_and_poster` |
| 8 | `site/prerender.py`: `full=True` | `test_the_prerender_script_is_its_pinned_program` |
| 8 | `site/_quarto.yml`: the `resources` entries removed | `test_quarto_runs_the_script_and_ships_the_media` |
| 9 | `_replay_embed_violations` is test code and has no mutant; its cases are exercised by the page mutants of Task 10 | |
| 10 | `site/story/into-the-plane.qmd`: ` autoplay` added to the light video | `test_no_page_has_unpinned_video_markup[site/story/into-the-plane.qmd]`; `test_manifest_matches_the_source[story/into-the-plane.qmd]` |
| 10 | `site/index.qmd`: the description div removed | `test_no_page_has_unpinned_video_markup[site/index.qmd]`; `test_description_numbers_are_the_archives[index.qmd:demo_2d]`; `test_the_two_2d_descriptions_are_the_same` |
| 10 | `site/story/two-hidden-masses.qmd`: the two embeds, with their divs, exchanged | `test_no_page_has_unpinned_video_markup[site/story/two-hidden-masses.qmd]`; `test_manifest_matches_the_source[story/two-hidden-masses.qmd]` |
| 10 | `site/story/how-many-masses.qmd`: a second, hand-written `<video src="../media/demo_1d.light.mp4" controls></video>` | `test_no_page_has_unpinned_video_markup[site/story/how-many-masses.qmd]` |
| 10 | the 1D description: `x = 2.48` changed to `x = 2.47` | `test_description_numbers_are_the_archives[story/the-search-in-one-dimension.qmd:demo_1d]` |
| 10 | the echolocation description: the sentence "After 12 readings the cloud has settled." added | `test_description_numbers_are_the_archives[story/gravitational-echolocation.qmd:demo_echolocation_3d]` |
| 10 | the model description: "tenth" changed to "fifth" | `test_description_numbers_are_the_archives[story/how-many-masses.qmd:demo_model_comparison]` |
| 10 | the manifest: `Fig("clip", 500, "demo_multi_mass")` given the height 450 | `test_manifest_matches_the_source[story/two-hidden-masses.qmd]`; `test_each_clip_is_embedded_at_its_declared_height` |
| 10 | `site/assets/demo_2d.gif` restored from `main` | `test_retired_visuals_are_gone_but_recorded[demo_2d.gif]` |
| 10 | `README.md`: the line `![2D inference demo](assets/demo_2d.gif)` restored | `test_no_page_mentions_a_retired_asset[README.md]`; `test_readme_points_to_the_site_and_describes_replay_data` |
| 10 | the Reproducibility page: "K=2: 0.3990" changed to "K=2: 0.4990" | `test_current_prose_identifies_corrected_assets_and_honest_model_results` |
| 10 | the Reproducibility page: "The demo GIFs are the exceptions: they are committed images, described in the next section." put back | `test_deploy_sentence_says_how_each_figure_is_redrawn` |
| 10 | the 1D page: "which shows a sample of five hundred" removed | `test_the_1d_page_says_how_many_dots_are_shown` |
| 11 | `demo_2d.py`: `TRUE_M = 0.15` | `test_truth_sits_inside_its_prior_box[scenario-demo_2d]`; `test_archive_matches_its_scenario[demo_2d]`; `test_gate_inputs[demo_2d]`; `test_gate_summaries[demo_2d]` (these three measured, Q7); `test_the_plane_cell_uses_the_demos_truth` |
| 11 | `assets/replays/demo_2d.npz` restored from `main` | `test_truth_sits_inside_its_prior_box[archive-demo_2d]`; `test_archive_matches_its_scenario[demo_2d]`; `test_gate_inputs[demo_2d]`; `test_gate_summaries[demo_2d]`; `test_description_numbers_are_the_archives[index.qmd:demo_2d]`, `[story/into-the-plane.qmd:demo_2d]` |
| 11 | `site/story/into-the-plane.qmd`: the cell's mass put back to 0.15 | `test_the_plane_cell_uses_the_demos_truth` |
| 11 | the same page: "tight knot at the truth" put back | `test_the_plane_cells_knot_is_offset_by_about_its_width` |
| 12 | Getting Started: the sentence "The 400–411 block was then run exactly once;" put back | `test_run_once_is_stated_as_protocol[site/reproduce/getting-started.qmd]`; `test_the_protocol_sentences` |
| 12 | the Reproducibility page: the link to the annealed-jitter design removed | `test_the_earlier_seed_blocks_are_disclosed_with_their_records` |
| 14 | `run_command`: the encoder looked up after `run(args)` returns | `test_video_without_an_encoder_stops_before_the_run` (each clip demo) |
| 14 | `run_command`: `EncoderError` not caught | `test_a_failed_clip_keeps_the_archive_and_png` |
| 14 | `add_common_arguments`: `--output` kept as an alias of `--output-dir` | `test_output_is_no_longer_an_option` (each demo) |
| 14 | `write_outputs`: `style("light")` whatever `theme` says | `test_theme_dark_draws_the_dark_png` (each demo) |
| 14 | `write_outputs`: `save_png` given `plt.figure(figsize=fig.get_size_inches())`, a blank figure of the same size | `test_png_is_the_final_frame` (each clip demo); `test_density_command_saves_the_figure_it_drew`; `test_a_one_observation_frame_is_not_blank` does not fail (it calls `final_frame` itself) |
| 14 | `write_outputs`: the directory not created | `test_demo_main_writes_its_archive_and_png` (each demo) |
| 14 | `demo_1d._run_demo`: `n_particles` not passed to `record` | `test_command_prints_the_recorded_run[demo_1d]`; `test_demo_main_writes_its_archive_and_png[demo_1d]` |
| 14 | `print_estimate`: the ESS line's denominator read from a module constant of 1000 | `test_command_prints_the_recorded_run` (the five) |
| 14 | `demo_model_comparison._run_demo`: `record` called a second time for the printout | `test_model_comparison_command_prints_from_the_replay_and_records_once` |
| 14 | `demo_density`: `add_video_argument` called | `test_density_has_no_video_option`; `test_demo_parser_defaults_and_overrides[demo_density]` does not fail (it does not look at `video` for that demo) |
| 14 | `README.md`: the `demo-2d` line's comment put back to `# → output/demo_2d.gif` | `test_command_lists_name_the_files_the_commands_write[README.md]` |
| 15 | `clocks.viz`: `render_clip` removed from `__all__` | `test_viz_exports_render_clip`; `test_viz_exports_exactly_the_documented_names` |
| 15 | `pyproject.toml`: the version put back to `0.1.0`, the lock left | `test_the_version_is_0_2_0_in_the_project_and_the_lock` |
| 15 | `README.md`: the upgrade note's `--output` bullet removed | `test_the_readme_has_the_upgrade_note` |
| 15 | the Architecture page: a node `animate["_animate"]` put back | `test_the_architecture_page_names_only_modules_that_exist` |

What the table rests on. The three existing failures of the `TRUE_M` row were measured (Q7); every other expected set is derived from the tests as this plan specifies them, and from the existing tests as read. Three kinds of row are the least certain and are named so that a difference there is read as a finding about the plan, not about the code: rows that say "list them from the run", where the set depends on how many fixtures share a builder; rows on a shared part, whose per-clip cases depend on which clips the implementer routes through it; and the `_style.py` row, which assumes `save_png` and `frame_rgba` share one function as Task 2 requires. Four behaviours have no mutant, each for a stated reason: the encoder's standard error going to a file (a blocked pipe hangs the suite instead of failing it); the layout rectangles' numbers (decision D20; the overlap and containment test catches a broken table); the wheel entry-point check (a script run by CI, not a test); and the contact sheets.

- [ ] **Step 1: Run each mutation** and record, for each, the exact set of failing test IDs and any change in what is collected.
- [ ] **Step 2: Report** the table of results. Where the failing set differs from the expected set, say which tests were extra or missing; do not change tests or code to make the table match. A difference is a finding for the coordinator.

---

### Hand-over

The coordinator opens the pull request as a draft after Task 1 and, in its description, gives: Task 1's "Probe results" (the sheets, the timings here and on the runner, Quarto 1.11.9's items (a) to (g), the encoder's version line on the runner, and the format decision); the final contact sheets with Task 13's findings and what is left for a person (Firefox, Safari, byte ranges on the deployed site); the render times of Task 8; the printed estimate of the re-recorded 2D archive; the mutation table of Task 16; and the upgrade note. After review converges and the pull request is marked ready, the coordinator confirms on that head, by SHA and with a non-empty `conclusion` for each: CI's default and slow steps and the wheel check; the site check on Quarto 1.11.9, with `Figure contract OK: 10 pages, 18 figures checked.` in its log and the pre-render step's twelve `rendered` lines; and all three replay-report jobs, each with `demo_2d: reproduced` or, if not, the report's first difference quoted in the pull request (the gate, not the report, decides). After the merge, the deployed landing page is opened once: the poster shows and the clip plays.

## Self-review

**Section 11, item 3, against the tasks.** Every clip drawn and timed, a clean build on Quarto 1.11.9, cold-runner timings: Task 1. `_video.py`: Task 2. The clip builders: Tasks 3 to 6. The encoder dependency: Task 2. The demo commands: Task 14. The migration note: Task 15. The 2D truth with its archive, clip, story cell and the box-margin guard: Task 11. The pre-render script: Task 8. Embeds and descriptions: Task 10. Clip support in the check: Task 7; in the guard: Tasks 9 and 10. The rest of the provenance prose: Tasks 10, 12 and 14. Removal of the GIFs: Task 10; of `_animate.py`, the old panel functions and their tests: Task 15. "If drawing shows the archive lacks something": Task 1, Step 7.

**Obligations elsewhere in the design.** Section 3: `EXPECTED_VISUAL_HASHES` loses the rest of its rows (Task 10); the calibration report's dated note (Task 10). Section 4: rebuild on change, the encoder's path and version printed, the lookup order, pixel sizes asserted, native controls, one description per clip, the stylesheet rule (Tasks 2, 3, 7, 8, 10). Section 5: every convention and each of the six clips (Tasks 3 to 6), drawn before any layout number is final (Task 1). Section 7: the commands, the break and its note, the wheel check, README and Getting Started, the Architecture page and `docs/someday-maybe.md` (Tasks 14 and 15). Section 8: every bullet (Task 11). Section 9: the replay-data paragraph, the echolocation and model-comparison sentences, the CI paragraph (Task 10); "run exactly once" (Task 12); the vocabulary and the calibration link's label were done in pull request 2. Section 10: the clip rows of the manifest, the clip checks, the decoded stream, the manifest against the source, retired assets (Tasks 7, 9, 10); the path filter already holds `assets/replays/**` and the study JSON. "Testing": the clip tests (Tasks 2 to 6 and 8), the site fixtures (Task 7), `tests/test_cli.py` (Task 14), the pinned sentences and descriptions (Tasks 10 to 12). "Mutation checks": Task 16. "Checks by eye": Tasks 1 and 13.

**Names across tasks.** `Clip.show(row, azimuth=None)`, `Shot(row, repeat, azimuth)`, `shots`, `build_clip`, `final_frame`, `render_clip(replay, path, *, theme, encoder) -> int`, `frame_rgba`, `encode(frames, path, *, width, height, encoder) -> int`, `probe -> StreamInfo`, `dwell_frames`, `find_encoder`, `CLIP_HEIGHTS`, `BUILDERS`, `CLIP_DEMOS`, `render_site_media(archives, media, *, full, demos, sources, report)`, `Embed`, `REPLAY_EMBEDS`, `_embed_block`, `Fig("clip", height, name)` are used with these signatures wherever they appear. Inset copies are named `"<inset gid>-<artist gid>"` in Tasks 3 and 4.

**Each task's tests at its own commit.** Tasks 3 to 6 parametrize over `BUILDERS`, so a clip's shared tests appear with its builder. Task 7's fixtures need Task 2 only. Task 9's page tests are vacuous until Task 10 fills the table. Task 10's description test reads the 0.15 archive, and Task 11 changes archive and text together. Task 14's tests of `--video` need Tasks 2 to 6.

**The site after each task that touches it.** Task 8: pages unchanged, media built and copied, unreferenced. Task 10: clips embedded, manifest and guard in step, GIFs gone, prose true. Task 11: the 2D clip, descriptions and cell agree with the new truth. Tasks 12, 14, 15: prose only. At no commit does a page name a file that is not there, and from Task 8 on every renderer is called by the build.

