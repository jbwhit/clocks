import json, sys
from multiprocessing import Pool
from pathlib import Path
from clocks._scenarios import run_echolocation_3d, ECHO_SWEEP_RANGES

CELL = (float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))


def job(a):
    seed, rr = a
    r = run_echolocation_3d(
        seed, rr, ess_target=CELL[0], rejuvenation_steps=CELL[1], proposal_scale=CELL[2]
    )
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}


if __name__ == "__main__":
    dev = json.loads(Path(sys.argv[1]).read_text())
    ref = {
        (x["seed"], x["range_r"]): x
        for x in dev["results"]
        if (x["ess_target"], x["rejuvenation_steps"], x["proposal_scale"]) == CELL
    }
    jobs = sorted(ref)
    with Pool(8) as p:
        out = p.map(job, jobs)
    maxrel = 0
    flips = []
    ev = 0
    for r in out:
        x = ref[(r["seed"], r["range_r"])]
        if r["passed"] != x["passed"]:
            flips.append((r["seed"], r["range_r"], x["passed"], r["passed"]))
        ev += r["forward_model_evaluations"] != x["forward_model_evaluations"]
        for k in ("mean", "std"):
            for a, b in zip(r[k], x[k]):
                maxrel = max(maxrel, abs(a - b) / max(abs(b), 1e-300))
    print(
        "cell",
        CELL,
        "n",
        len(out),
        "pass head",
        sum(r["passed"] for r in out),
        "pass dev",
        sum(x["passed"] for x in ref.values()),
    )
    print("flips", flips, "eval mismatches", ev, "max rel diff", maxrel)
