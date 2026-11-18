import subprocess, sys, shutil
from pathlib import Path

W = Path("/Users/jonathan/projects/clocks/.worktrees/audit-evidence")
ECHO = "site/story/gravitational-echolocation.qmd"
REP = "docs/2026-08-16-development-calibration.md"
REPRO = "site/reproduce/reproducibility.qmd"
MUTANTS = [
    (
        "M1 echo page overall 46/72 -> 49/72",
        ECHO,
        "passed 46/72 cases overall",
        "passed 49/72 cases overall",
    ),
    (
        "M2 echo page close 12/12 -> 11/12",
        ECHO,
        "The close range passed 12/12.",
        "The close range passed 11/12.",
    ),
    (
        "M3 echo page gate factor 20 -> 10",
        ECHO,
        "frozen factor\n20 gate",
        "frozen factor\n10 gate",
    ),
    (
        "M4 echo page coverage diag 12/12 far -> 11/12 far",
        ECHO,
        "12/12 far-range three",
        "11/12 far-range three",
    ),
    (
        "M5 report dev per-range 12,12,12,8,4,1 -> 12,12,12,9,4,1",
        REP,
        "12, 12, 12, 8, 4, and 1",
        "12, 12, 12, 9, 4, and 1",
    ),
    ("M6 report dev ratio 66.226 -> 61.226", REP, "66.226", "61.226"),
    (
        "M7 report multi cert 12/12 -> 11/12",
        REP,
        "The multi-mass cell passed 12/12 certification cases",
        "The multi-mass cell passed 11/12 certification cases",
    ),
    (
        "M8 report dev selected 49/72 -> 50/72",
        REP,
        "ties the best pass count at 49/72",
        "ties the best pass count at 50/72",
    ),
    ("M9 repro page replays claim", REPRO, "run exactly once", "run three times"),
]
TESTS = [
    "tests/test_generated_assets.py",
    "tests/test_site_prose_contract.py",
    "tests/test_calibration.py",
]


def clean():
    for p in W.rglob("__pycache__"):
        if ".venv" not in p.parts:
            shutil.rmtree(p, ignore_errors=True)
    s = subprocess.run(
        ["git", "status", "--porcelain"], cwd=W, capture_output=True, text=True
    ).stdout
    assert s.strip() == "", f"dirty tree: {s}"


clean()
env = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PATH": __import__("os").environ["PATH"],
    "HOME": __import__("os").environ["HOME"],
}
base = subprocess.run(
    ["uv", "run", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
    cwd=W,
    capture_output=True,
    text=True,
    env=env,
)
print("BASELINE:", base.stdout.strip().splitlines()[-1])
for name, rel, old, new in MUTANTS:
    path = W / rel
    orig = path.read_text()
    assert orig.count(old) == 1, (name, orig.count(old))
    try:
        path.write_text(orig.replace(old, new))
        clean_cache = [
            shutil.rmtree(p, ignore_errors=True)
            for p in W.rglob("__pycache__")
            if ".venv" not in p.parts
        ]
        r = subprocess.run(
            ["uv", "run", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
            cwd=W,
            capture_output=True,
            text=True,
            env=env,
        )
        print(f"{name}: {r.stdout.strip().splitlines()[-1]}")
    finally:
        path.write_text(orig)
    clean()
