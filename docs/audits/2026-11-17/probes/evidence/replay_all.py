import json, sys, math
from multiprocessing import Pool
from pathlib import Path
from clocks._scenarios import run_echolocation_3d, ECHO_SWEEP_RANGES


def job(a):
    seed, rr = a
    r = run_echolocation_3d(
        seed, rr, ess_target=0.9, rejuvenation_steps=1, proposal_scale=1.5
    )
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}


if __name__ == "__main__":
    cert = json.loads(Path(sys.argv[1]).read_text())
    jobs = [(s, r) for r in ECHO_SWEEP_RANGES for s in range(400, 412)]
    with Pool(8) as p:
        out = p.map(job, jobs)
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
    ref = {(x["seed"], x["range_r"]): x for x in cert["results"]}
    maxrel = 0.0
    flips = []
    evals_diff = 0
    for r in out:
        x = ref[(r["seed"], r["range_r"])]
        if r["passed"] != x["passed"] or r["covered_3sigma"] != x["covered_3sigma"]:
            flips.append((r["seed"], r["range_r"]))
        if r["forward_model_evaluations"] != x["forward_model_evaluations"]:
            evals_diff += 1
        for k in ("mean", "std"):
            for a, b in zip(r[k], x[k]):
                maxrel = max(maxrel, abs(a - b) / max(abs(b), 1e-300))
    print("pass/coverage flips:", flips)
    print("eval-count mismatches:", evals_diff, "of", len(out))
    print("max relative diff in mean/std:", maxrel)
    print("passed total:", sum(r["passed"] for r in out))
