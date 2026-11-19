import json, statistics, sys, collections

d = json.load(open(sys.argv[1]))
rs = d["results"]
print("n results", len(rs))
by = collections.defaultdict(list)
for r in rs:
    by[r["range_r"]].append(r)
tot = sum(r["passed"] for r in rs)
print("passed", tot, "/", len(rs))
for k in sorted(by):
    g = by[k]
    print(
        k,
        "pass",
        sum(r["passed"] for r in g),
        "/",
        len(g),
        "cov3",
        sum(r["covered_3sigma"] for r in g),
        "median pos_std",
        statistics.median(r["pos_std"] for r in g),
        "seeds",
        sorted(r["seed"] for r in g)[:3],
        "...",
    )
near, far = min(by), max(by)
mn = statistics.median(r["pos_std"] for r in by[near])
mf = statistics.median(r["pos_std"] for r in by[far])
print("far/near median pos_std ratio", mf / mn)
# recompute pass from tolerances
t = d["tolerances"]
bad = [
    r
    for r in rs
    if r["passed"]
    != (
        r["position_error"] <= t["position_error_max"]
        and r["mass_error"] <= t["mass_error_max"]
    )
]
print("pass flag inconsistent with tolerances:", len(bad))
for r in bad[:5]:
    print(r["seed"], r["range_r"], r["position_error"], r["mass_error"], r["passed"])
