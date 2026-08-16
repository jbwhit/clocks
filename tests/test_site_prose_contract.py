import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_site_links.py"
_SPEC = importlib.util.spec_from_file_location("check_site_links", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
check_site_links = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_site_links)


def test_current_prose_contract_scans_scripts_but_not_archives(tmp_path: Path) -> None:
    current_script = tmp_path / "scripts" / "demo.py"
    archived_plan = tmp_path / "docs" / "plans" / "old.md"
    current_script.parent.mkdir(parents=True)
    archived_plan.parent.mkdir(parents=True)
    current_script.write_text('"""Old jitter instructions."""\n', encoding="utf-8")
    archived_plan.write_text("jitter is historical\n", encoding="utf-8")

    assert check_site_links._prose_claim_failures(tmp_path) == [
        "scripts/demo.py: 'jitter' (tempered SMC replaced uncorrected perturbations)"
    ]
