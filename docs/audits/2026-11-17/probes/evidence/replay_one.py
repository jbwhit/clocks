import json, sys, time
from pathlib import Path
from clocks._scenarios import run_echolocation_3d

cert = json.loads(Path(sys.argv[1]).read_text())
pairs = [(int(a), float(b)) for a, b in (p.split(":") for p in sys.argv[2:])]
for seed, rr in pairs:
    t = time.time()
    r = run_echolocation_3d(
        seed, rr, ess_target=0.9, rejuvenation_steps=1, proposal_scale=1.5
    )
    dt = time.time() - t
    ref = next(x for x in cert["results"] if x["seed"] == seed and x["range_r"] == rr)
    print(f"seed={seed} r={rr} time={dt:.1f}s")
    for k in (
        "mean",
        "std",
        "position_error",
        "mass_error",
        "forward_model_evaluations",
        "passed",
    ):
        a = r[k]
        a = a.tolist() if hasattr(a, "tolist") else a
        print(f"  {k}: head={a}\n  {' ' * len(k)}  json={ref[k]}  equal={a == ref[k]}")
