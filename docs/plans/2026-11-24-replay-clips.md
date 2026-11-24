# Replay Clips Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw the six demo animations when the site is built, as H.264 clips with posters in both themes, from the committed replay archives; switch the pages, the provenance prose and the demo commands to them; lower the 2D demo's true mass to 0.10; and remove the committed GIFs with the code that made them.

**Architecture:** `clocks._video` turns raw frames into an H.264 file and reads one back; it imports no Matplotlib. The `clocks._clip_*` modules build, from a replay and a palette, a figure with fixed axes whose artists are set to one archive row at a time; `clocks._clips.render_clip` walks a replay's shots through the encoder. `clocks._site_media.render_site_media`, called by a three-line Quarto pre-render script, writes twelve clips and twelve posters to the git-ignored `site/media/`. Pages embed them as raw-HTML blocks that the page-source guard pins whole and the rendered-site check decodes; the demo commands record a run, print it from the replay and write an archive and a PNG, with the clip on request.

**Tech Stack:** Python 3.12+, NumPy, Matplotlib 3.10 (declared `>=3.9`), FFmpeg through `imageio-ffmpeg` (a development dependency) or `ffmpeg` on `PATH`, pytest, uv, ruff, Quarto 1.11.9 on the runner (1.9.32 on the planning machine), GitHub Actions.

**Spec:** docs/plans/2026-11-19-replay-and-animation-design.md

**Status:** Draft. Not yet reviewed.

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
- **D3. Sixteen tasks, one plan, with a split point after Task 13.** After Task 13 the site is correct, every renderer is called by the site's pre-render step, the GIFs are gone and the 2D truth is 0.10; the commands still write GIFs through `_animate.py`, and the README and Getting Started still say so, truthfully. Tasks 14 and 15 (commands, public names, version, upgrade note, removal of the old modules) and a second mutation run could be a fourth pull request. If the pull request is split there, Task 16's table is split by the task column.
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
- **D21. Descriptions use the trace headers' decimals** (`clocks._figures.trace_decimals`: two for positions, three for masses), so a description, a header and a poster agree digit for digit. The commands print one more decimal than that (three for positions, as today, and four for masses), because review of a re-recording rests on the printed estimates (design, "Risks").

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
- **Q2. Encoding and decoding.** 296 synthetic frames of 1500 by 900, piped as raw frames with the settings of Global Constraints, through each build: exit 0, 27,454 bytes each, **different bytes** between the two builds; boxes `ftyp`, `moov`, `free`, `mdat`. Decoding either file with either build (`-hide_banner -nostdin -i <file> -map 0:v:0 -f null - -progress pipe:1 -nostats`): the same line, `Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1500x900, 12 kb/s, 20 fps, 20 tbr, 10240 tbn (default)`, and a last progress line `frame=296`. The file cut to half its bytes: exit 0, the same boxes and stream line, `frame=129`, with `Invalid NAL unit size` on standard error. An output path in a directory that does not exist: the encoder exits and the next frame write raises `BrokenPipeError`. Small clips, both builds: three solid frames of 64 by 48 written 2, 1 and 4 times decode to `frame=7` with the same boxes; copied with `-c copy` and no flag the boxes are `ftyp`, `free`, `mdat`, `moov` and the count is still 7; sixty frames of seeded noise make 117,153 bytes, and the half file decodes to `frame=29` with its stream line intact; a text file named `.mp4` gives exit 183 and no `Video:` line; re-encoded with `-c:v mpeg4`, and with `-c:v libx264 -pix_fmt yuv444p`, the lines are those quoted in Task 2's parser test. `-version` prints `ffmpeg version ...` as the first line of standard output.
- **Q3. Frames.** Inside `style(theme)`: `plt.figure(figsize=(7.5, 4.5), layout="none")` reports no layout engine; after one `savefig` it reports a `ConstrainedLayoutEngine`, and a second save after `add_axes` warns "There are no gridspecs with layoutgrids". `plt.figure(figsize=(7.5, 4.5))` followed by `fig.set_layout_engine("none")` reports a `PlaceHolderLayoutEngine`, the same object after four saves and a draw, with no warning under `warnings.simplefilter("error")` and with a plain axes, an inset and a 3D axes all keeping their positions. `fig.savefig(buffer, format="rgba", dpi=200)` gives 1500 by 840, 900, 1000, 1080 and 1100 pixels for heights of 4.2, 4.5, 5.0, 5.4 and 5.5 inches, every alpha 255, equal pixel for pixel to the file `save_png` writes. A 500-dot figure with two axes redraws and saves in about 20 ms; a 3D axes with 500 dots in about 32 ms. `ax.inset_axes(...)` is in `ax.child_axes` and not in `fig.axes`. A styled canvas with nothing on it has 1 distinct colour; with one empty axes about 218; with a status line about 447.
- **Q4. The dwell schedule**, by the rule of Global Constraints: 296 frames for T = 50 and 356 for T = 80; `[24, 50]` for T = 1; `[24, 18, 50]` for T = 2.
- **Q5. The camera**, by the design's formula with F = 306 (T = 80): frames 0, 76, 152, 229 and 305 have azimuths −75.0, −69.17850576840875, −55.103002582480016, −40.82149423159126 and −35.0. A constant-speed turn would give −65.03 at frame 76. With F = 24 (T = 1): frames 0, 6, 12 and 23 give −75.0, −68.65106286437309, −53.63515173270658 and −35.0.
- **Q6. The 2D run at M = 0.10**, in a scratch clone with `TRUE_M = 0.10` and `scripts/record_replays.py --out assets/replays --overwrite demo_2d`: final mean (1.41692, −1.22229, 0.10002) ± (0.11108, 0.11479, 0.00124), the design's M2; after observation 1, (3.630, −3.361, 0.1225) ± (1.465, 1.502, 0.0152); after 8, (1.861, −1.352, 0.0993) ± (0.304, 0.298, 0.0032). Largest |x–y correlation| after the prior: row 3, −0.690. Running mean of slowing: largest over the run 0.031689, largest in the last row 0.024270, 2 of 400 clock-rows above 1.1 times the latter. Zoom half-widths by the design's rule: 0.38156 (x), 0.56483 (y), 0.0031185 (M). Weak-field strength 0.0468. With that archive, `uv run pytest -q -m '' tests/test_replays.py tests/test_demo_records.py tests/test_shipped_ground_truths.py tests/test_record_replays_script.py`: 237 passed. The "Into the Plane" cell at 0.10: (1.3716, −1.0980, 0.1003) ± (0.1724, 0.1696, 0.0018), largest particle mass 0.1068, 0.162 from the true position; today, at 0.15: (1.4049, −1.0768, 0.1487) ± (0.1114, 0.1140, 0.0010), largest particle mass 0.1500.
- **Q7. `TRUE_M` put back to 0.15 against the 0.10 archive**, whole suite (`uv run pytest -q -m ''`): 3 failed, 1,918 passed. The three: `tests/test_replays.py::test_archive_matches_its_scenario[demo_2d]`, `::test_gate_inputs[demo_2d]`, `::test_gate_summaries[demo_2d]`.
- **Q8. The committed archives**, from `load_replay`. Final mean ± standard deviation at the trace headers' decimals: `demo_1d` (T 50, 1,000 particles, 3 clocks) x 2.48 ± 0.04, M 0.101 ± 0.001; `demo_2d` at 0.15 (T 50, 2,000, 8 clocks) x 1.43 ± 0.07, y −1.15 ± 0.08, M 0.149 ± 0.001; `demo_multi_mass` (T 80, 3,000, 5 clocks) x₁ −2.00 ± 0.04, x₂ 4.55 ± 0.05, M₁ 0.045 ± 0.001, M₂ 0.030 ± 0.001, and after observation 1 x₂ 6.61 ± 1.04, after 8 x₂ 4.62 ± 0.18; `demo_multi_mass_2d` (T 80, 4,000, 10 clocks) x₁ −2.38 ± 0.42, y₁ 2.03 ± 0.22, x₂ 4.13 ± 0.80, y₂ −0.95 ± 0.63, M₁ 0.052 ± 0.004, M₂ 0.028 ± 0.003; `demo_echolocation_3d` (T 80, 6,000, 27 clocks) x 0.98 ± 0.02, y 1.46 ± 0.02, z 2.93 ± 0.03, M 0.078 ± 0.002, `view_bounds` (−1.3, 4.8), `weight_in_view[0]` 0.0045 (printed to one decimal as 0.4%), smallest later value 0.9999999999999998, dots inside the view in rows 0, 1 and 80: 2, 500 and 500. `demo_model_comparison` (T 80, 2,000 per candidate): `model_ks` 1, 2, 3; prior one third each; the leader is K = 3 from observation 1, K = 2 from 10, K = 3 from 19; final (0, 0.39904, 0.60096). Position standard deviations after observation 1 against the prior's: `demo_1d` 0.27 against 4.92; echolocation 0.16 to 0.26 against 9.2 to 9.3. The running mean of slowing is positive in every clock-row of `demo_1d`, `demo_2d` and `demo_multi_mass` (smallest 0.0102).
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

- [ ] **Step 6: Record the results in this plan.** Add a section "Probe results" after "Hand-over", with: the local and runner Quarto versions and items (a) to (g) for each; the encoder's version line on the runner; per clip and theme, drawings, seconds and bytes, here and on the runner, and the two totals; the final `LAYOUTS`, by reference to the script; every change made to decisions D17 to D20; and the run's URL. The same numbers go to the coordinator for the pull request's description. Check each sheet's metadata as pull request 2 did: `Image.open(path).getexif()` is empty, `set(Image.open(path).info) == {"jfif", "jfif_density", "jfif_unit", "jfif_version"}`, and `[marker for marker, _ in Image.open(path).applist] == ["APP0"]`.

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
  - `test_video_module_imports_only_the_standard_library`: every module named by an `Import` or `ImportFrom` node in the top-level statements of `src/clocks/_video.py` has its first component in `sys.stdlib_module_names`. Only this catches `import imageio_ffmpeg` at module level, which would make `clocks.viz` fail to import from a wheel installed without the encoder.
  - `test_encode_then_probe`: the small clip; `encode` returns 7, `tmp_path` holds exactly `clip.mp4`, and `probe(path) == StreamInfo("h264", "yuv420p", 64, 48, 20.0, 7, True)`. Only this catches a wrong codec, pixel format, rate or `+faststart` flag in the command. No byte of the file is compared.
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
  - `test_parse_stream_line` (IDs and inputs): `ffmpeg_9` the line of Q2 → `("h264", "yuv420p", 1500, 900, 20.0)`; `with_colour_and_aspect` `"  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1500x1100 [SAR 1:1 DAR 15:11], 330 kb/s, 20 fps, 20 tbr, 10240 tbn (default)"` → `("h264", "yuv420p", 1500, 1100, 20.0)`; `mpeg4` (measured, Q2) `"  Stream #0:0[0x1](und): Video: mpeg4 (Simple Profile) (mp4v / 0x7634706D), yuv420p, 64x48 [SAR 1:1 DAR 4:3], 5 kb/s, 20 fps, 20 tbr, 10240 tbn (default)"` → `("mpeg4", "yuv420p", 64, 48, 20.0)`; `yuv444p` (measured, Q2) `"  Stream #0:0[0x1](und): Video: h264 (High 4:4:4 Predictive) (avc1 / 0x31637661), yuv444p(progressive), 64x48, 19 kb/s, 20 fps, 20 tbr, 10240 tbn (default)"` → `("h264", "yuv444p", 64, 48, 20.0)`; `fractional_rate` the first with `29.97 fps` → 29.97. The second and the last are constructed, not measured. Only this catches a pattern that takes `High` for the pixel format, the hexadecimal tag for the size, or `tbr` for the rate.
  - `test_parse_stream_line_refuses_other_text` (IDs `audio_only` `"  Stream #0:0: Audio: aac, 44100 Hz"`, `no_rate` the Q2 line with `, 20 fps` removed): `ValueError` whose message contains the offending text. Only this catches a build's rewording being read as a pass.
  - `test_box_order` on bytes built in the test with `struct.pack(">I4s", size, kind) + payload`: `ftyp`, `moov`, `free`, `mdat` → those four; a 64-bit box (`size` 1, then `struct.pack(">Q", 24)`) followed by `moov` → both; a last box whose stated size runs past the end → still listed. Only this catches a walk that stops at, or loops on, the special sizes.
  - `test_probe_counts_the_frames_a_truncated_clip_decodes_to`: sixty frames of 64 by 48 seeded noise (`np.random.default_rng(0).integers(0, 256, (48, 64, 4), np.uint8)` with alpha set to 255, a fresh draw per frame), each written once; the file cut to half its bytes. `probe` returns `codec == "h264"`, `moov_first is True` and `0 < frames < 60`. The exact count depends on the build and is not asserted. Only this catches a frame count read from the container's tables, which a cut file keeps whole.
  - `test_probe_sees_moov_after_mdat`: the small clip copied by the encoder without the flag (`[find_encoder(), "-loglevel", "error", "-y", "-i", clip, "-c", "copy", other]`): `probe(other).moov_first is False` and `frames == 7`. Only this catches `moov_first` hard-wired to true.
  - `test_probe_refuses_a_file_that_is_not_video`: a text file named `clip.mp4` raises `ValueError` matching `no video stream`. Only this catches a probe that returns defaults for an unreadable file.
  - `test_encoder_version_is_the_first_line`: `encoder_version(find_encoder())` starts with `ffmpeg version `. Only this catches a version line taken from standard error or left empty.

  In `tests/test_style.py`:

  - `test_frame_rgba_is_what_save_png_writes` (IDs `plain`, and `hostile` under `matplotlib.rc_context({"savefig.bbox": "tight", "savefig.transparent": True, "savefig.facecolor": "magenta"})`): a figure made inside `with style("light"):` with `plt.subplots(figsize=(7.5, 4.5))` and one line; after that block, `frame_rgba(fig)` has `1500 * 900 * 4` bytes; reshaped to (900, 1500, 4) it equals `np.round(imread(path, format="png") * 255)` for the file `save_png(fig, path)` wrote in the same case; and its first pixel is the light background with alpha 255. Only this catches frames at another resolution, cropped, transparent or on another background than the poster.

  In `tests/test_video.py` also: `test_the_encoder_is_a_development_dependency_only`: with `tomllib`, some entry of `pyproject.toml`'s `dependency-groups.dev` starts with `imageio-ffmpeg`, and no entry of `project.dependencies` starts with `imageio`. Only this catches the encoder becoming a requirement of the installed library.

- [ ] **Step 2: Stage the names, then run.** Create `_video.py` with the classes, `FPS`, `dwell_frames` returning `[24] * (n + 1)`, `find_encoder` returning `shutil.which("ffmpeg") or ""`, `encode` writing nothing and returning 0, `parse_stream_line` returning `("", "", 0, 0, 0.0)`, `box_order` returning `[]`, `probe` returning a `StreamInfo` of those, `encoder_version` returning `""`; add `frame_rgba` returning `b""`. Run `uv run pytest tests/test_video.py tests/test_style.py -q`: every new test fails on its assertion, except `test_video_module_imports_only_the_standard_library`, which passes already.

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
  - `test_axes_never_move` (each clip): after `fig.canvas.draw()` at rows 0, 1, 4 and 9, the mapping of every axes gid in `all_axes(fig)` to `(ax.get_position().bounds, ax.get_xlim(), ax.get_ylim())` (and `get_zlim()` where there is one) equals the mapping at row 0; no gid is `None` and none repeats. Only this catches autoscaling in a middle frame.
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
  - `test_2d_clock_fills_follow_the_running_mean` (each theme), on `committed("demo_2d")`, with `running = running_slowing(observations)` and `largest = np.nanmax(running)` computed in the test: at row 0 every face has alpha 0 and every edge is `ink`; at the last row clock c's face is `background + (running[-1, c] / largest) * (ink - background)` (`atol=1e-9`), and the largest of those ratios is below 1 (asserted first: it is 0.848 today and 0.766 at M = 0.10, Q6 and Q8); at the row and clock of `np.nanargmax(running)` the face is `ink`. Only this catches a scale taken from the last row's largest mean, under which the darkest final clock would be `ink`.
  - `test_2d_a_reading_above_one_gives_a_background_fill`: on a copy of `small("demo_2d")` with `observations[0, 2] = 1.02`, at row 1 clock 2's face is the background (`atol=1e-12`), under `warnings.simplefilter("error")`. Only this catches an unclipped blend on a real figure.
  - `test_2d_labels`: the map holds exactly one visible text reading `stronger fill = slower clock`, in `ink`. Only this catches the key being dropped.
  - `test_viz_exports_render_clip`: `clocks.viz.render_clip is clocks._clips.render_clip`, and the name is in `clocks.viz.__all__`.

  In `tests/test_site_figure_style.py`: `LIBRARY_DRAWING_MODULES` gains `"src/clocks/_clip_parts.py"`, `"src/clocks/_clip_maps.py"` and `"src/clocks/_clips.py"`; new `test_every_clip_module_is_scanned`: every `src/clocks/_clip*.py` is in `LIBRARY_DRAWING_MODULES` (only this catches a later clip module escaping the colour check); new `test_clip_modules_import_no_model_code`: for each `src/clocks/_clip*.py`, `_video.py` and (from Task 8) `_site_media.py`, the modules it imports from `clocks` are among `clocks._replay`, `clocks._style`, `clocks._figures`, `clocks._video` and the `clocks._clip*` modules (only this catches a renderer calling the forward model).

- [ ] **Step 2: Stage the names, then run.** Create the three modules with the names of Interfaces: the pure functions raising `NotImplementedError`, `build_2d` returning `Clip(new_figure(450), lambda row, azimuth: None)` with `new_figure` implemented, `shots` returning `[]`, `final_frame` returning the built clip's figure, `render_clip` returning 0; add the helpers and tables to the two test helper modules. Run `uv run pytest tests/test_clips.py tests/test_site_figure_style.py -q`: the literal tests fail on `NotImplementedError`, the artist tests in `one(...)` on missing gids, `test_a_one_observation_frame_is_not_blank` on 1 distinct value, the shot and encode tests on their counts; the size test, `test_the_layout_engine_stays_off`, `test_panels_are_on_the_canvas_and_apart` and the two guard tests already pass.

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

<!-- more tasks follow -->
