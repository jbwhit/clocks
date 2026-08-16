import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_site_links.py"
_REPO_ROOT = _SCRIPT.parents[1]
_SPEC = importlib.util.spec_from_file_location("check_site_links", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
check_site_links = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_site_links)


def test_current_prose_contract_scans_scripts_but_not_archives(tmp_path: Path) -> None:
    benign_script = tmp_path / "scripts" / "instrument.py"
    stale_script = tmp_path / "scripts" / "demo.py"
    archived_plan = tmp_path / "docs" / "plans" / "old.md"
    benign_script.parent.mkdir(parents=True)
    archived_plan.parent.mkdir(parents=True)
    benign_script.write_text(
        '"""Measure physical oscillator jitter.\n\n'
        "Historical comparison: retired resampling jitter changed targets.\n"
        '"""\n',
        encoding="utf-8",
    )
    stale_script.write_text('"""Set jitter_tau for inference."""\n', encoding="utf-8")
    archived_plan.write_text("jitter_tau is historical\n", encoding="utf-8")

    assert check_site_links._prose_claim_failures(tmp_path) == [
        "scripts/demo.py: 'jitter_tau' (tempered SMC replaced annealed jitter)"
    ]


def test_repository_current_prose_contract_is_clean() -> None:
    assert check_site_links._prose_claim_failures(_REPO_ROOT) == []
