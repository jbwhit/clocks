import json, sys
from multiprocessing import Pool
from pathlib import Path
from clocks._scenarios import run_multi_mass_2d


def job(seed):
    r = run_multi_mass_2d(
        seed, ess_target=0.7, rejuvenation_steps=2, proposal_scale=3.0
    )
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}


if __name__ == "__main__":
    cert = json.loads(Path(sys.argv[1]).read_text())
    ref = {x["seed"]: x for x in cert["results"]}
    with Pool(8) as p:
        out = p.map(job, sorted(ref))
    maxrel = 0
    for r in out:
        x = ref[r["seed"]]
        for a, b in zip(r["mean"] + r["std"], x["mean"] + x["std"]):
            maxrel = max(maxrel, abs(a - b) / max(abs(b), 1e-300))
    print(
        "pass head",
        sum(r["passed"] for r in out),
        "flips",
        [r["seed"] for r in out if r["passed"] != ref[r["seed"]]["passed"]],
        "eval mismatches",
        sum(
            r["forward_model_evaluations"]
            != ref[r["seed"]]["forward_model_evaluations"]
            for r in out
        ),
        "maxrel",
        maxrel,
    )
