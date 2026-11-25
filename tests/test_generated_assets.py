"""Contracts for the corrected generated evidence shipped by Task 11."""

import importlib.util
import json
import re
import subprocess
from hashlib import sha256
from pathlib import Path
from statistics import median

import pytest

from clocks._replay import load_replay

ROOT = Path(__file__).parents[1]
ASSETS = ROOT / "assets"
SITE_ASSETS = ROOT / "site/assets"
# The rendered-site check, loaded by path as tests/test_site_figure_style.py
# loads it, for its REPLAY_CSS.
_CHECKER = ROOT / "scripts" / "check_site_figures.py"
_CHECKER_SPEC = importlib.util.spec_from_file_location("check_site_figures", _CHECKER)
assert _CHECKER_SPEC is not None and _CHECKER_SPEC.loader is not None
check_site_figures = importlib.util.module_from_spec(_CHECKER_SPEC)
_CHECKER_SPEC.loader.exec_module(check_site_figures)

RETIRED_VISUALS = {
    "demo_density.png": (
        "a7a6e6e9628640ac08e89152b8471d37db981c264380ba567634032a0f59dc7a"
    ),
    "echolocation_range_study.png": (
        "6d62a1c88c78837299bb434e66574d6930eef24c6cf9805e42ec66e141dd86bc"
    ),
    "demo_1d.gif": "417ae3523e95e85e91feca7f67e2a8bc7347006883ea27b1101f0cc164477483",
    "demo_2d.gif": "eb581e327d950d7bedce87f28c383c35aa38e99792291d5ace2291ae090e1a3b",
    "demo_multi_mass.gif": (
        "bb2a3a2a74270114a516133cf2b8ef9484eb1bbc2d40b542d42d54013e30f8ad"
    ),
    "demo_multi_mass_2d.gif": (
        "ecce3ca3187010d73f5ea651826f43a1e2057ad24a45f8ad040d14979cdaa967"
    ),
    "demo_model_comparison.gif": (
        "27af99630edf6578e635d60cfd2f085442f8429ca008ce25afc935b823c6c9f2"
    ),
    "demo_echolocation_3d.gif": (
        "5df30d7c052c3e366216866c9b0ecc991cc777467515dd1c7699de743413d34f"
    ),
}
ECHO_CERT_SOURCE_HASH = (
    "a4c6b1b7c3c2fce273aaa19f01289e6ca34de2ed8d633f827e798e1af4f47941"
)


def test_certified_range_json_matches_raw_hash_and_tracked_records() -> None:
    shipped_path = ASSETS / "echolocation_range_study.json"
    tracked_path = ROOT / "docs/calibration/echolocation_range_certification.json"
    shipped_bytes = shipped_path.read_bytes()
    shipped = json.loads(shipped_bytes)
    tracked = json.loads(tracked_path.read_text())

    assert sha256(shipped_bytes).hexdigest() == ECHO_CERT_SOURCE_HASH
    assert shipped["seed_block"] == 400
    assert shipped["seed_role"] == "protected"
    assert shipped["seeds"] == list(range(400, 412))
    assert len(shipped["results"]) == 72
    assert shipped["results"] == tracked["results"]
    assert tracked["source"] == {
        "format": "schema_v1",
        "sha256": ECHO_CERT_SOURCE_HASH,
    }


def _certified_range_summary() -> tuple[str, str, str]:
    """Pass counts by range, overall total, and far/close median std ratio."""
    study = json.loads(
        (ROOT / "docs/calibration/echolocation_range_certification.json").read_text()
    )
    runs_by_range = [
        [run for run in study["results"] if run["range_r"] == range_r]
        for range_r in study["ranges"]
    ]
    counts = [sum(run["passed"] for run in runs) for runs in runs_by_range]
    close_std = median(run["pos_std"] for run in runs_by_range[0])
    far_std = median(run["pos_std"] for run in runs_by_range[-1])
    return (
        ", ".join(str(count) for count in counts),
        f"{sum(counts)}/{len(study['results'])}",
        f"{far_std / close_std:.3f}",
    )


def test_current_prose_identifies_corrected_assets_and_honest_model_results() -> None:
    report = (ROOT / "docs/2026-08-16-development-calibration.md").read_text()
    echo_story = " ".join(
        (ROOT / "site/story/gravitational-echolocation.qmd").read_text().split()
    )
    model_story = (ROOT / "site/story/how-many-masses.qmd").read_text()
    provenance = " ".join(
        (ROOT / "site/reproduce/reproducibility.qmd").read_text().split()
    )
    current_asset_pages = [
        ROOT / "README.md",
        ROOT / "site/index.qmd",
        ROOT / "site/reproduce/reproducibility.qmd",
        ROOT / "site/story/the-search-in-one-dimension.qmd",
        ROOT / "site/story/into-the-plane.qmd",
        ROOT / "site/story/two-hidden-masses.qmd",
        ROOT / "site/story/how-many-masses.qmd",
        ROOT / "site/story/beyond-point-masses.qmd",
        ROOT / "site/story/gravitational-echolocation.qmd",
    ]

    assert "a1b016b" in report
    # The exact protocol record lives on the Reproducibility page; the story
    # pages state the results in plain words and link there. The test
    # searches whitespace-normalized prose for the passages below. The pass
    # counts, the overall total and the std ratio are computed from the
    # certification artifact; the 12/12 figures and factor 20 are literals.
    pass_counts, total, std_ratio = _certified_range_summary()
    assert pass_counts == "12, 12, 11, 8, 3, 0"
    assert (
        f"Pass counts out of twelve, from closest to farthest: {pass_counts}. "
        in echo_story
    )
    assert (
        f"The frozen cell passed {total} cases overall; pass counts from close "
        f"to far were {pass_counts}. The close range passed 12/12. The far/close "
        f"median position-standard-deviation ratio was {std_ratio}, above the "
        "frozen factor 20 gate, and the 12/12 far-range three-standard-deviation "
        "coverage is retained as a diagnostic rather than promoted after "
        "inspection into a new gate."
    ) in provenance
    assert (
        "It does not estimate population reliability or justify a universal "
        "pass probability."
    ) in provenance
    model = load_replay(ASSETS / "replays/demo_model_comparison.npz")
    meta = model.meta
    ks = [int(k) for k in model.arrays["model_ks"]]
    final = model.arrays["model_posterior"][-1]
    p2, p3 = final[ks.index(2)], final[ks.index(3)]
    assert (
        "The animation on [How Many Masses?](../story/how-many-masses.qmd) replays "
        f"the packaged default run: {meta['n_observations']} observations, "
        f"{meta['n_particles']:,} particles per candidate, and seed {meta['seed']}. "
    ) in provenance
    assert (
        f"With K={meta['true_k']} truth, its final estimate is K=2: {p2:.4f} and "
        f"K=3: {p3:.4f}. "
    ) in provenance
    assert (
        "The executed cell on that page is a separate 25-observation, "
        "400-particle calculation that ends near K=2: 0.740 and K=3: 0.260. "
    ) in provenance
    assert (
        "A different finite data horizon changes the evidence supplied by the "
        "noisy realization, and a different finite-particle budget changes the "
        "Monte Carlo estimate of that evidence."
    ) in provenance
    assert "as the verdict shows" not in model_story
    for page in current_asset_pages:
        text = page.read_text().casefold()
        assert "pre-remediation" not in text
        assert "pending task 11" not in text
        assert "asset regeneration remains pending" not in text


PROVENANCE_VOCABULARY = (
    "committed gif",
    "committed png",
    "packaged default",
    "one-shot",
    "certif",
    "seed-block 400",
    "400–411",
    "population reliability",
    "committed replay",
    "replay file",
)


@pytest.mark.parametrize(
    "page",
    sorted([ROOT / "site/index.qmd", *(ROOT / "site/story").glob("*.qmd")]),
    ids=lambda page: page.name,
)
def test_story_pages_leave_provenance_to_the_reproducibility_page(
    page: Path,
) -> None:
    text = " ".join(page.read_text().casefold().split())

    for phrase in PROVENANCE_VOCABULARY:
        assert phrase not in text, f"{page.name} carries provenance text: {phrase!r}"


def test_landing_page_links_to_the_figure_provenance_section() -> None:
    landing = (ROOT / "site/index.qmd").read_text()
    provenance = (ROOT / "site/reproduce/reproducibility.qmd").read_text()

    assert "reproduce/reproducibility.qmd#sec-figures" in landing
    assert "{#sec-figures}" in provenance


@pytest.mark.parametrize(
    ("name", "expected_hash"), RETIRED_VISUALS.items(), ids=list(RETIRED_VISUALS)
)
def test_retired_visuals_are_gone_but_recorded(name: str, expected_hash: str) -> None:
    report = (ROOT / "docs/2026-08-16-development-calibration.md").read_text()

    assert not (ASSETS / name).exists()
    assert not (SITE_ASSETS / name).exists()
    assert f"| `{name}` | `{expected_hash}` |" in report.splitlines()
    assert re.search(
        rf"`{re.escape(name)}` was removed from both trees on \d{{4}}-\d{{2}}-\d{{2}};",
        report,
    )


def _reproducibility_page() -> str:
    return (ROOT / "site/reproduce/reproducibility.qmd").read_text()


def test_density_provenance_names_its_run() -> None:
    meta = load_replay(ROOT / "assets/replays/demo_density.npz").meta
    page = " ".join(_reproducibility_page().split())

    assert (
        "**Replay data.** Each animation, and the density figure on "
        "[Beyond Point Masses](../story/beyond-point-masses.qmd), is drawn when the "
        "site is built, from `assets/replays/<name>.npz`, the recorder's archive of "
        "that demo's packaged run. "
    ) in page
    assert (
        f"For the density demo that is {meta['n_observations']} observations, "
        f"{meta['n_particles']:,} particles and seed {meta['seed']}. "
    ) in page
    assert "density PNG" not in page


def test_the_cloud_size_in_the_prose_is_the_archives() -> None:
    meta = load_replay(ROOT / "assets/replays/demo_density.npz").meta
    size = meta["cloud_size"]
    reproducibility = " ".join(_reproducibility_page().split())
    story = " ".join((ROOT / "site/story/beyond-point-masses.qmd").read_text().split())
    caption = f"from {size} particles sampled from the final posterior"

    assert f"a {size}-particle sample for display" in reproducibility
    assert story.count(caption) == 2
    for archive in sorted((ASSETS / "replays").glob("*.npz")):
        replay = load_replay(archive)
        if "cloud" in replay.arrays:
            assert replay.meta["cloud_size"] == 500, archive.name
    model = load_replay(ASSETS / "replays/demo_model_comparison.npz")
    assert "cloud" not in model.arrays
    assert (
        "An archive of a filter run holds the configuration, seed and observations, "
        "a summary of the cloud before any observation and after each, and a "
        "500-particle sample for display. The model-comparison archive holds no "
        "cloud: it records each candidate's posterior probability, evidence and "
        "effective sample size after each observation, and each candidate's final "
        "estimate."
    ) in reproducibility


def test_range_study_provenance_names_its_source() -> None:
    page = " ".join(_reproducibility_page().split())

    assert (
        "The figure is drawn when the site is built, from the certified "
        "block-400 records in `assets/echolocation_range_study.json`. "
        "The frozen cell passed "
    ) in page
    assert "range-study PNG" not in page


def test_deploy_sentence_says_how_each_figure_is_redrawn() -> None:
    page = " ".join(_reproducibility_page().split())

    assert (
        "so every figure and animation is redrawn at deploy time: most figures by "
        "running the model again from source, and the density figure, the "
        "range-study figure and the six animations from committed data files."
    ) in page
    assert "demo GIFs" not in page
    assert "committed images" not in page
    assert "regenerated from source at deploy time" not in page
    assert "range-study PNG" not in page


def test_provenance_section_describes_figures_drawn_at_build_time() -> None:
    raw = _reproducibility_page()
    page = " ".join(raw.split())

    assert "## Where the figures come from {#sec-figures}" in raw
    assert (
        "[development calibration report]"
        "(https://github.com/jbwhit/clocks/blob/main/"
        "docs/2026-08-16-development-calibration.md) "
        "is the record of the August 2026 release"
    ) in page


def test_site_check_description_names_the_figure_data() -> None:
    page = " ".join(_reproducibility_page().split())
    assert (
        "touch the site, the library, or the data the site's figures are drawn from"
        in page
    )


def test_the_seed_sentence_covers_only_cells_that_draw_random_numbers() -> None:
    page = " ".join(_reproducibility_page().split())
    assert (
        "Every demo, and every executed cell on this site that draws random "
        "numbers, fixes its RNG seed."
    ) in page


def test_the_ci_bullets_link_exactly_the_workflow_files() -> None:
    raw = _reproducibility_page()
    page = " ".join(raw.split())
    section = raw.split("Four workflows run on GitHub Actions:")[1]
    section = section.split("\n## ")[0]
    links = re.findall(
        r"^- \[`([\w-]+\.yml)`\]\((\S+)\)",
        section,
        re.MULTILINE,
    )
    files = {path.name for path in (ROOT / ".github/workflows").glob("*.yml")}

    assert "Four workflows run on GitHub Actions:" in page
    assert len(links) == 4
    assert len({text for text, _ in links}) == 4
    base = "https://github.com/jbwhit/clocks/blob/main/.github/workflows/"
    assert all(target == f"{base}{text}" for text, target in links)
    assert {text for text, _ in links} == files


def test_the_replay_report_bullet_claims_only_what_the_workflow_guarantees() -> None:
    page = " ".join(_reproducibility_page().split())
    assert (
        "It is a diagnostic: every step is allowed to fail without failing the run."
    ) in page
    assert "never fails a pull request" not in page


_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
_NUMBER_WORDS |= {7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven"}
_NUMBER_WORDS |= {12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen"}


def test_the_range_caption_counts_the_studys_runs_and_ranges() -> None:
    study = json.loads((ASSETS / "echolocation_range_study.json").read_text())
    ranges = study["ranges"]
    per_range = {
        sum(run["range_r"] == range_r for run in study["results"]) for range_r in ranges
    }
    assert len(per_range) == 1
    phrase = (
        f"{_NUMBER_WORDS[per_range.pop()]} runs at each of "
        f"{_NUMBER_WORDS[len(ranges)]} ranges"
    )
    lines = (ROOT / "site/story/gravitational-echolocation.qmd").read_text()
    captions = [
        line for line in lines.splitlines() if line.startswith('#|   - "Resolution')
    ]

    assert len(captions) == 2
    for caption in captions:
        assert phrase in caption
        assert (
            "the median posterior standard deviation (for position, the three "
            "coordinates' standard deviations combined in quadrature);"
        ) in caption


def test_video_files_are_called_build_products() -> None:
    page = " ".join(_reproducibility_page().split())
    assert (
        "The video files are build products: they are not committed, and their "
        "bytes are not pinned."
    ) in page


def test_echolocation_animation_provenance() -> None:
    page = " ".join(_reproducibility_page().split())
    assert (
        "**Echolocation animation.** The animation places the mass at two "
        "circumradii along a fixed off-axis direction, and is one fixed-seed "
        "visualization at the closest, most favorable range, not a reliability "
        "measurement."
    ) in page
    for stale in (
        "will change when the assets are next regenerated",
        "not a re-render of the current code",
        "committed GIF",
    ):
        assert stale not in page


def test_readme_describes_the_site_and_the_commands_as_they_are() -> None:
    """Only this catches the README pointing at a deleted image, or claiming of
    the commands what is true only of the site."""
    readme = " ".join((ROOT / "README.md").read_text().split())
    for sentence in (
        "The site animates six of the demos and draws the density figure, all "
        "from replay data",
        "The site's clips and posters are build products and are not committed.",
        "What is committed is replay data, `assets/replays/<demo>.npz`: what each "
        "packaged run produced, in numbers, tied to the code by tests of its "
        "inputs and of bounded summaries.",
        "Six of the commands above still write GIFs, drawn by the earlier "
        "animation code; `demo-density` writes a PNG.",
    ):
        assert sentence in readme
    assert "assets/demo_2d.gif" not in readme
    assert "commits no rendered media" not in readme


def test_the_commands_are_described_as_they_are() -> None:
    """Only this catches the page implying that the commands produce the clips.

    Pull request 4 replaces the sentence and this test.
    """
    page = " ".join(_reproducibility_page().split())
    assert (
        "`uv run demo-density` records the same configured scenario and draws its "
        "result. The other six `uv run demo-*` commands run the same scenarios "
        "again and still write GIFs, drawn by the earlier animation code and not "
        "from these archives."
    ) in page


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _block(lines: list[str], header: str) -> list[str]:
    """The lines under the one line ``header``, up to the next non-blank line
    indented no deeper than it (blank lines are kept)."""
    assert lines.count(header) == 1, header
    start = lines.index(header) + 1
    depth = _indent(header)
    end = start
    while end < len(lines) and (not lines[end].strip() or _indent(lines[end]) > depth):
        end += 1
    return lines[start:end]


def test_quarto_runs_the_script_and_ships_the_media() -> None:
    """Only this catches media drawn but never copied to the site, or the
    stylesheet dropped from the list Quarto links for HTML.

    The file is read as text (the tests have no YAML reader: decision D45), and
    each pin is anchored to its block, so the same lines under another key do
    not pass.
    """
    lines = (ROOT / "site" / "_quarto.yml").read_text().split("\n")

    project = _block(lines, "project:")
    assert "  pre-render: prerender.py" in project
    resources = _block(project, "  resources:")
    assert resources == ['    - "media/*.mp4"', '    - "media/*.png"']

    html = _block(_block(lines, "format:"), "  html:")
    css = [index for index, line in enumerate(html) if line.startswith("    css:")]
    assert len(css) == 1
    at = css[0]
    assert html[at] == "    css:"
    assert html[at + 1 : at + 3] == ["      - styles.css", "      - replay.css"]
    assert not html[at + 3].lstrip().startswith("-")
    assert sum(line.lstrip().startswith("css:") for line in lines) == 1


def test_replay_css_is_the_canonical_rule() -> None:
    """Only this catches the rule edited, extended or commented out in the source."""
    rule = b"figure.replay video { max-width: 100%; height: auto; }\n"
    assert (ROOT / "site" / "replay.css").read_bytes() == rule
    assert rule == check_site_figures.REPLAY_CSS


def test_someday_maybe_no_longer_says_static_gifs() -> None:
    """Only this catches a live note still describing the site as it was."""
    note = " ".join((ROOT / "docs" / "someday-maybe.md").read_text().split())
    assert "static GIFs" not in note
    assert "pre-rendered video drawn from committed replay data" in note


# The five current-facing places that once said the reserved block was "run
# exactly once" (decision D33). The repository records the run; it cannot
# enforce that there was only one, so each place states the rule as protocol.
_PROTOCOL_FILES = (
    "site/reproduce/reproducibility.qmd",
    "site/reproduce/getting-started.qmd",
    "site/method/the-particle-filter.qmd",
    "README.md",
    "docs/someday-maybe.md",
)
# Records of what was decided or attested at the time, which keep the phrase.
_HISTORICAL_PREFIXES = ("docs/audits/", "docs/superpowers/", "docs/plans/")
_HISTORICAL_FILES = {"docs/2026-08-16-development-calibration.md"}
_SPEC_URL = "https://github.com/jbwhit/clocks/blob/main/"
# Each earlier seed block's design record, with the phrase that supports the
# disclosure sentence citing it.
_EARLIER_BLOCK_RECORDS = {
    "docs/superpowers/specs/2026-07-03-clone-freeze-diagnosis.md": (
        "holdout seeds 100–111"
    ),
    "docs/superpowers/specs/2026-07-02-annealed-jitter-design.md": (
        "seeds 200-211, run exactly once"
    ),
    "docs/superpowers/specs/2026-07-19-3d-echolocation-design.md": (
        "seeds 300-311, run once"
    ),
}


def _normalised(relative: str) -> str:
    return " ".join((ROOT / relative).read_text(encoding="utf-8").split())


@pytest.mark.parametrize("relative", _PROTOCOL_FILES, ids=_PROTOCOL_FILES)
def test_run_once_is_stated_as_protocol(relative: str) -> None:
    """Only this catches the attestation coming back in any of the five places."""
    text = _normalised(relative).casefold()
    assert "exactly once" not in text
    assert "by protocol" in text


def test_only_historical_records_still_say_exactly_once() -> None:
    """Only this catches the phrase returning in a current-facing document that
    is not one of the five (decision D33)."""
    tracked = subprocess.run(
        ["git", "ls-files", "*.md", "*.qmd"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    saying_it = [
        relative
        for relative in tracked
        if relative and "exactly once" in _normalised(relative).casefold()
    ]
    assert saying_it, "the historical records were expected to keep the phrase"
    current = [
        relative
        for relative in saying_it
        if not relative.startswith(_HISTORICAL_PREFIXES)
        and relative not in _HISTORICAL_FILES
    ]
    assert current == []


def test_the_protocol_sentences() -> None:
    expected = {
        "site/reproduce/reproducibility.qmd": (
            "By protocol each reserved block is for one run: block 400–411 was run "
            "under the frozen settings, using the scenario-default cell via "
            "`--seed-block 400`, and protected blocks reject control overrides. "
            "The repository records that run; it does not enforce that there was "
            "only one."
        ),
        "site/reproduce/getting-started.qmd": (
            "By protocol the 400–411 block is reserved for one run, made under "
            "those frozen settings;"
        ),
        "README.md": (
            "which by protocol is for one run and was run under those frozen settings."
        ),
        "site/method/the-particle-filter.qmd": (
            "and the reserved 400–411 block, which by protocol is for one run, was "
            "run under those frozen settings."
        ),
    }
    for relative, sentence in expected.items():
        assert sentence in _normalised(relative), relative


def test_the_earlier_seed_blocks_are_disclosed_with_their_records() -> None:
    """Only this catches a disclosure whose citation has gone."""
    page = " ".join(_reproducibility_page().split())
    assert (
        "Earlier implementations used three earlier blocks, as their design "
        "records state: seeds 100–111 as a holdout"
    ) in page
    for relative, phrase in _EARLIER_BLOCK_RECORDS.items():
        assert f"]({_SPEC_URL}{relative})" in page, relative
        assert (ROOT / relative).is_file(), relative
        assert phrase in _normalised(relative), relative
