"""Contracts for the corrected generated evidence shipped by Task 11."""

import json
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
ASSETS = ROOT / "assets"
SITE_ASSETS = ROOT / "site/assets"

EXPECTED_VISUAL_HASHES = {
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
    "demo_density.png": (
        "a7a6e6e9628640ac08e89152b8471d37db981c264380ba567634032a0f59dc7a"
    ),
    "demo_echolocation_3d.gif": (
        "5df30d7c052c3e366216866c9b0ecc991cc777467515dd1c7699de743413d34f"
    ),
    "echolocation_range_study.png": (
        "6d62a1c88c78837299bb434e66574d6930eef24c6cf9805e42ec66e141dd86bc"
    ),
}
ECHO_CERT_SOURCE_HASH = (
    "a4c6b1b7c3c2fce273aaa19f01289e6ca34de2ed8d633f827e798e1af4f47941"
)


@pytest.mark.parametrize(("name", "expected_hash"), EXPECTED_VISUAL_HASHES.items())
def test_corrected_visual_release_bytes_are_complete_and_mirrored(
    name: str, expected_hash: str
) -> None:
    root_path = ASSETS / name
    site_path = SITE_ASSETS / name

    assert root_path.is_file() and root_path.stat().st_size > 0
    assert site_path.is_file() and site_path.stat().st_size > 0
    root_bytes = root_path.read_bytes()
    assert site_path.read_bytes() == root_bytes
    assert sha256(root_bytes).hexdigest() == expected_hash
    if name.endswith(".gif"):
        assert root_bytes[:6] in (b"GIF87a", b"GIF89a")
    else:
        assert root_bytes.startswith(b"\x89PNG\r\n\x1a\n")


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


def test_current_prose_identifies_corrected_assets_and_honest_model_results() -> None:
    report = (ROOT / "docs/2026-08-16-development-calibration.md").read_text()
    echo_story = (ROOT / "site/story/gravitational-echolocation.qmd").read_text()
    model_story = (ROOT / "site/story/how-many-masses.qmd").read_text()
    normalized_model_story = " ".join(model_story.split())
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
    for expected_hash in EXPECTED_VISUAL_HASHES.values():
        assert expected_hash in report
    assert "12, 12, 11, 8, 3, 0" in echo_story
    assert "12/12" in echo_story
    assert "66.524" in echo_story
    assert "12/12 far-range" in echo_story
    assert "population reliability" in echo_story
    assert "K=2: 0.3990" in model_story
    assert "K=3: 0.6010" in model_story
    assert "K=2: 0.740" in model_story
    assert "K=3: 0.260" in model_story
    assert "finite data horizon" in normalized_model_story
    assert "finite-particle" in model_story
    assert "as the verdict shows" not in model_story
    for page in current_asset_pages:
        text = page.read_text().casefold()
        assert "pre-remediation" not in text
        assert "pending task 11" not in text
        assert "asset regeneration remains pending" not in text
