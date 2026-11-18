import json, statistics, re, collections
from pathlib import Path

R = Path("/Users/jonathan/projects/clocks")
doc = (R / "docs/2026-08-16-development-calibration.md").read_text()


def cells(path):
    d = json.loads(path.read_text())
    g = collections.defaultdict(list)
    for r in d["results"]:
        g[(r["ess_target"], r["rejuvenation_steps"], r["proposal_scale"])].append(r)
    return d, g


def lines(g):
    out = []
    ranked = []
    for (e, s, p), c in sorted(g.items()):
        n = sum(r["passed"] for r in c)
        me = statistics.median(r["normalized_error"] for r in c)
        mv = statistics.median(r["forward_model_evaluations"] for r in c)
        out.append(
            f"ess={e:.2f} steps={s} scale={p:.2f}: {n}/{len(c)}, median normalized error={me:.3f}, median forward evaluations={mv:.0f}"
        )
        ranked.append(((-n, me, mv, s), (e, s, p)))
    ranked.sort()
    return out, ranked


for name in ("multi_mass_2d", "echolocation_range"):
    d, g = cells(R / f"docs/calibration/{name}_development.json")
    out, ranked = lines(g)
    missing = [l for l in out if l not in doc]
    print(name, "dev lines", len(out), "missing from doc:", missing)
    print("  winner", ranked[0], "runner-up", ranked[1])
    w = g[ranked[0][1]]
    if name == "multi_mass_2d":
        print("  failing seeds", [r["seed"] for r in w if not r["passed"]])
    else:
        by = collections.defaultdict(list)
        for r in w:
            by[r["range_r"]].append(r)
        print("  per-range pass", [sum(r["passed"] for r in by[k]) for k in sorted(by)])
        print(
            "  far cov",
            sum(r["covered_3sigma"] for r in by[8.0]),
            "ratio",
            statistics.median(r["pos_std"] for r in by[8.0])
            / statistics.median(r["pos_std"] for r in by[2.0]),
        )
        print(
            "  close max pos/mass err",
            max(r["position_error"] for r in by[2.0]),
            max(r["mass_error"] for r in by[2.0]),
        )
for name in ("multi_mass_2d", "echolocation_range"):
    d, g = cells(R / f"docs/calibration/{name}_certification.json")
    c = next(iter(g.values()))
    print(
        name,
        "cert pass",
        sum(r["passed"] for r in c),
        "/",
        len(c),
        "median normerr",
        statistics.median(r["normalized_error"] for r in c),
        "median evals",
        statistics.median(r["forward_model_evaluations"] for r in c),
    )
    if name == "echolocation_range":
        close = [r for r in c if r["range_r"] == 2.0]
        print(
            "  close max pos/mass",
            max(r["position_error"] for r in close),
            max(r["mass_error"] for r in close),
        )
