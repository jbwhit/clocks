"""How much of the echolocation range/mass information comes from the
sqrt(1+2Phi) nonlinearity (common-mode leak) rather than field curvature?

Compare Cramer-Rao bounds for the contrast observable Q @ f(Phi) with
  sqrt : f = sqrt(1 + 2 Phi)          (what the code simulates and fits)
  lin  : f = 1 + Phi                  (pure geometry: Q @ Phi; also what
                                       Q @ log(rate) gives if rate = exp(Phi),
                                       i.e. what frequency *ratios* carry when
                                       the 2nd-order term is the 1PN one)
Also: size of the O(Phi^2) model ambiguity in the contrasts relative to noise.
"""

import numpy as np

from clocks._scenarios import (
    ECHO_DIRECTION,
    ECHO_M_TRUE,
    ECHO_N_OBSERVATIONS,
    ECHO_NOISE_STD,
    ECHO_R_HEAD,
    ECHO_SWEEP_RANGES,
    build_head_lattice,
    contrast_matrix,
)

head = build_head_lattice().positions
Q = contrast_matrix(len(head))


def phi(theta: np.ndarray) -> np.ndarray:
    pos, m = theta[:3], theta[3]
    return -m / np.linalg.norm(head - pos, axis=1)


MODELS = {
    "sqrt": lambda p: np.sqrt(1 + 2 * p),
    "lin": lambda p: 1 + p,
    "exp": lambda p: np.exp(p),  # 1PN-consistent second order, rate differences
    "1pn": lambda p: np.sqrt(1 + 2 * p + 2 * p**2),
}


def jac(f, theta):
    eps = 1e-7
    cols = []
    for k in range(4):
        d = np.zeros(4)
        d[k] = eps * max(1.0, abs(theta[k]))
        cols.append((Q @ f(phi(theta + d)) - Q @ f(phi(theta - d))) / (2 * d[k]))
    return np.column_stack(cols)


def crb(f, theta):
    J = jac(f, theta)
    F = J.T @ J * ECHO_N_OBSERVATIONS / ECHO_NOISE_STD**2
    cov = np.linalg.inv(F)
    # transform position to (range R) via gradient of |pos|
    pos = theta[:3]
    g = np.zeros(4)
    g[:3] = pos / np.linalg.norm(pos)
    sR = np.sqrt(g @ cov @ g)
    sM = np.sqrt(cov[3, 3])
    corr = (g @ cov[:, 3]) / (sR * sM)
    return sR, sM, corr


print("range_r  R     |2Phi|max  model  sd(R)    sd(M)    corr(R,M)")
for rr in ECHO_SWEEP_RANGES:
    pos = ECHO_DIRECTION * rr * ECHO_R_HEAD
    theta = np.r_[pos, ECHO_M_TRUE]
    p = phi(theta)
    for name in ("sqrt", "lin"):
        sR, sM, c = crb(MODELS[name], theta)
        print(
            f"{rr:5.1f}  {rr * ECHO_R_HEAD:6.2f}  {np.max(np.abs(2 * p)):.4f}  "
            f"{name:5s} {sR:8.4f} {sM:8.5f} {c:+.4f}"
        )
    # model-ambiguity at second order: sqrt vs 1pn contrast difference
    diff = Q @ (MODELS["sqrt"](p) - MODELS["1pn"](p))
    eff_noise = ECHO_NOISE_STD / np.sqrt(ECHO_N_OBSERVATIONS)
    print(
        f"        |Q(sqrt-1pn)| = {np.linalg.norm(diff):.2e}  "
        f"(= {np.linalg.norm(diff) / eff_noise:.1f} x per-contrast noise after "
        f"{ECHO_N_OBSERVATIONS} obs);  |Q sqrt| = {np.linalg.norm(Q @ np.sqrt(1 + 2 * p)):.2e}"
    )
