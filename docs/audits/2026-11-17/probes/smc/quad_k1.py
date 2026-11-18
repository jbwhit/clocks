"""Compare SMC (public API) against brute-force quadrature for K=1, 1-D clocks.

Usage: uv run python quad_k1.py <case> <n_particles> <n_seeds>
"""

import sys

import numpy as np
from scipy.special import logsumexp

from clocks import (
    ClockArray,
    InferenceConfig,
    MassConfig,
    NoiseConfig,
    PriorConfig,
    SimulationConfig,
    build_particle_filter,
    simulate,
)

CASES = {
    # page case: sharp posterior
    "page": dict(clocks=[-5.0, 0.0, 5.0], d=1.0, x=2.5, m=0.10, sigma=0.005, n=25),
    # weak data: broad posterior, bimodal likely
    "weak": dict(clocks=[-5.0, 0.0, 5.0], d=1.0, x=2.5, m=0.10, sigma=0.02, n=2),
    # moderate
    "mid": dict(clocks=[-5.0, 0.0, 5.0], d=1.0, x=-1.0, m=0.04, sigma=0.01, n=4),
}
XR = (-8.0, 8.0)
MR = (0.005, 0.15)


def rates_grid(xs, ms, clocks, d):
    # xs, ms broadcastable arrays; returns (..., C) rates and validity
    c = np.asarray(clocks)
    r = np.sqrt((c - xs[..., None]) ** 2 + d * d)
    phi = -ms[..., None] / r
    valid = np.all(np.abs(phi) <= 0.05, axis=-1)
    return np.sqrt(1.0 + 2.0 * np.maximum(phi, -0.5)), valid


def loglik_grid(xs, ms, obs, clocks, d, sigma):
    f, valid = rates_grid(xs, ms, clocks, d)
    ybar = obs.mean(axis=0)
    n = obs.shape[0]
    within = np.sum((obs - ybar) ** 2)
    q = within + n * np.sum((f - ybar) ** 2, axis=-1)
    ll = -n * len(clocks) * np.log(sigma * np.sqrt(2 * np.pi)) - 0.5 * q / sigma**2
    return np.where(valid, ll, -np.inf), valid


def quadrature(obs, clocks, d, sigma):
    # coarse grid to find mass, then refine cells near the mode(s)
    nx, nm = 1600, 1600
    xe = np.linspace(*XR, nx + 1)
    me = np.linspace(*MR, nm + 1)
    xc = 0.5 * (xe[1:] + xe[:-1])
    mc = 0.5 * (me[1:] + me[:-1])
    X, M = np.meshgrid(xc, mc, indexing="ij")
    ll, valid = loglik_grid(X, M, obs, clocks, d, sigma)
    # validity fraction with a very fine grid (smooth boundary)
    xf = np.linspace(*XR, 8001)
    mf = np.linspace(*MR, 8001)
    xfc = 0.5 * (xf[1:] + xf[:-1])
    mfc = 0.5 * (mf[1:] + mf[:-1])
    pvalid = 0.0
    for chunk in np.array_split(xfc, 20):
        Xf, Mf = np.meshgrid(chunk, mfc, indexing="ij")
        _, v = rates_grid(Xf, Mf, clocks, d)
        pvalid += v.sum()
    pvalid /= len(xfc) * len(mfc)
    top = ll.max()
    sel = np.argwhere(ll > top - 60.0)
    # dilate selection by 2 cells
    mask = np.zeros_like(ll, dtype=bool)
    for di in range(-2, 3):
        for dj in range(-2, 3):
            ii = np.clip(sel[:, 0] + di, 0, nx - 1)
            jj = np.clip(sel[:, 1] + dj, 0, nm - 1)
            mask[ii, jj] = True
    cells = np.argwhere(mask)
    sub = int(min(24, max(1, np.sqrt(4e7 / len(cells)))))
    off = (np.arange(sub) + 0.5) / sub
    dx = (XR[1] - XR[0]) / nx
    dm = (MR[1] - MR[0]) / nm
    logw_all = []
    xs_all = []
    ms_all = []
    for block in np.array_split(cells, max(1, len(cells) * sub * sub // 2_000_000)):
        xs = (xe[block[:, 0]][:, None, None] + dx * off[None, :, None]) + 0 * off[
            None, None, :
        ]
        ms = (me[block[:, 1]][:, None, None] + dm * off[None, None, :]) + 0 * off[
            None, :, None
        ]
        l, _ = loglik_grid(xs, ms, obs, clocks, d, sigma)
        logw_all.append(l.ravel())
        xs_all.append(xs.ravel())
        ms_all.append(ms.ravel())
    lw = np.concatenate(logw_all)
    xs = np.concatenate(xs_all)
    ms = np.concatenate(ms_all)
    cell_area = dx * dm / sub**2
    box = (XR[1] - XR[0]) * (MR[1] - MR[0])
    log_z = logsumexp(lw) + np.log(cell_area / box) - np.log(pvalid)
    w = np.exp(lw - logsumexp(lw))
    mean = np.array([w @ xs, w @ ms])
    std = np.sqrt(np.array([w @ (xs - mean[0]) ** 2, w @ (ms - mean[1]) ** 2]))
    return log_z, mean, std, pvalid, len(cells)


def main():
    case = CASES[sys.argv[1]]
    n_particles = int(sys.argv[2])
    n_seeds = int(sys.argv[3])
    clock_array = ClockArray(
        positions=np.array([[c] for c in case["clocks"]]), track_offset=case["d"]
    )
    sim = simulate(
        SimulationConfig(
            clock_array=clock_array,
            ground_truth=MassConfig(np.array([[case["x"]]]), np.array([case["m"]])),
            noise=NoiseConfig(case["sigma"]),
            n_observations=case["n"],
            seed=42,
        )
    )
    obs = np.array([o.rates for o in sim.observations])
    log_z, mean, std, pvalid, ncell = quadrature(
        obs, case["clocks"], case["d"], case["sigma"]
    )
    print(
        f"quad: logZ={log_z:.4f} mean={mean} std={std} pvalid={pvalid:.5f} cells={ncell}"
    )
    zs = []
    mus = []
    sds = []
    for seed in range(n_seeds):
        pf = build_particle_filter(
            InferenceConfig(
                clock_array=clock_array,
                noise=NoiseConfig(case["sigma"]),
                prior=PriorConfig(XR, MR),
                n_particles=n_particles,
                n_masses=1,
                seed=1000 + seed,
            )
        )
        for o in sim.observations:
            pf.update(o)
        est = pf.estimate()
        zs.append(pf.log_evidence)
        mus.append(est["mean"])
        sds.append(est["std"])
        print(
            f"seed {seed}: logZ={pf.log_evidence:.4f} mean={est['mean']} "
            f"std={est['std']} ess={est['ess']:.0f}"
        )
    zs = np.array(zs)
    mus = np.array(mus)
    sds = np.array(sds)
    print(
        f"SMC mean-of-means={mus.mean(0)} (se {mus.std(0, ddof=1) / np.sqrt(len(mus))}) mean-of-stds={sds.mean(0)}"
    )
    print(
        f"SMC logZ mean={zs.mean():.4f} sd={zs.std(ddof=1):.4f} "
        f"se={zs.std(ddof=1) / np.sqrt(len(zs)):.4f}  diff={zs.mean() - log_z:+.4f}"
    )


if __name__ == "__main__":
    main()
