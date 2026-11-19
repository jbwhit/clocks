"""Density forward model: scalar (scipy quad) vs batch vs an independent
high-accuracy reference, and the sigma-A degeneracy claim."""

import warnings

import numpy as np
from scipy.integrate import quad
from scipy.special import erf

from clocks import ClockArray
from clocks.physics import (
    PhysicsDomainError,
    clock_rates_density_gaussian,
    clock_rates_density_gaussian_batch,
)


def reference_phi(mu, sigma, amp, xc, h, limit=10.0):
    """Independent reference: split at the clock and at mu, generous limits."""
    lo, hi = mu - limit * sigma, mu + limit * sigma
    pts = sorted({p for p in (xc, mu) if lo < p < hi})
    f = lambda x: amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.hypot(x - xc, h)
    edges = [lo, *pts, hi]
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        # further split around the clock peak at scale h
        val, _ = quad(f, a, b, limit=500, epsabs=0, epsrel=1e-13)
        total += val
    return -total


print("== scalar vs batch vs reference ==")
rng = np.random.default_rng(1)
worst_scalar = (0, None)
worst_batch = (0, None)
for trial in range(400):
    h = 10 ** rng.uniform(-3, 0.5)
    sigma = 10 ** rng.uniform(-1, 0.7)
    mu = rng.uniform(-6, 6)
    xc = rng.uniform(-8, 8)
    # choose amp so that |2 Phi| ~ 0.05 at most
    amp = 1.0
    ref1 = reference_phi(mu, sigma, 1.0, xc, h)
    amp = 0.02 / abs(ref1)
    ca = ClockArray(positions=np.array([[xc]]), track_offset=h)
    ref_rate = np.sqrt(1 + 2 * amp * ref1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            s = clock_rates_density_gaussian(np.array([mu, sigma, amp]), ca)[0]
            es = abs(s - ref_rate)
            if es > worst_scalar[0]:
                worst_scalar = (es, (mu, sigma, amp, xc, h))
        except PhysicsDomainError as e:
            pass
        try:
            b = clock_rates_density_gaussian_batch(np.array([[mu, sigma, amp]]), ca)[
                0, 0
            ]
            eb = abs(b - ref_rate)
            if eb > worst_batch[0]:
                worst_batch = (eb, (mu, sigma, amp, xc, h))
        except PhysicsDomainError:
            pass
print("worst scalar abs rate error:", worst_scalar)
print("worst batch  abs rate error:", worst_batch)

print("== targeted narrow-kernel case for scalar quad ==")
for h in (1e-1, 1e-2, 1e-3, 1e-4):
    mu, sigma = 0.0, 3.0
    for xc in (0.37, 4.1, 7.3):
        ref1 = reference_phi(mu, sigma, 1.0, xc, h)
        amp = 0.02 / abs(ref1)
        ca = ClockArray(positions=np.array([[xc]]), track_offset=h)
        try:
            s = clock_rates_density_gaussian(np.array([mu, sigma, amp]), ca)[0]
            rel = (1 - s) / (1 - np.sqrt(1 + 2 * amp * ref1)) - 1
            print(f"h={h:g} xc={xc}: scalar dip rel err {rel:+.2e}")
        except PhysicsDomainError as e:
            print(f"h={h:g} xc={xc}: scalar raised {e}")

print("== demo truth: scalar vs batch ==")
ca = ClockArray(
    positions=np.array([[-6.0], [-3.0], [0.0], [3.0], [6.0]]), track_offset=1.0
)
p = np.array([1.5, 2.0, 0.010])
s = clock_rates_density_gaussian(p, ca)
b = clock_rates_density_gaussian_batch(p[None, :], ca)[0]
print("scalar", s, "\nbatch ", b, "\nmax diff", np.max(np.abs(s - b)), " dip", 1 - s)

print("== Fisher for density demo (mu, sigma, A), noise 0.005, 60 obs ==")


def fwd(q):
    return clock_rates_density_gaussian_batch(q[None, :], ca)[0]


J = []
for k, eps in enumerate((1e-4, 1e-4, 1e-7)):
    d = np.zeros(3)
    d[k] = eps
    J.append((fwd(p + d) - fwd(p - d)) / (2 * eps))
J = np.column_stack(J)
F = J.T @ J * 60 / 0.005**2
cov = np.linalg.inv(F)
sd = np.sqrt(np.diag(cov))
print("CRB sd (mu, sigma, A):", sd)
print("corr matrix:\n", np.round(cov / np.outer(sd, sd), 4))
# total mass M = A sigma sqrt(2pi)
g = np.array([0.0, p[2], p[1]]) * np.sqrt(2 * np.pi)
print(
    "CRB sd of total mass:",
    np.sqrt(g @ cov @ g),
    "total mass",
    p[1] * p[2] * np.sqrt(2 * np.pi),
)
print("signal-to-noise per obs: max dip / sigma =", np.max(1 - s) / 0.005)
