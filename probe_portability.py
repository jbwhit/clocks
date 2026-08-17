"""Print bit-exact fingerprints of every primitive the population draw uses."""

import hashlib
import math
import platform
import sys
from decimal import Decimal, localcontext

import numpy as np

from clocks._reliability import generate_release_manifest


def h(values):
    return hashlib.sha256(
        "|".join(float(v).hex() for v in values).encode()
    ).hexdigest()[:16]


print(
    f"platform={platform.machine()} {platform.system()} "
    f"python={sys.version.split()[0]} numpy={np.__version__}"
)

randoms, normals, uniforms = [], [], []
lo, hi = math.log(2.0), math.log(2.5198420997897463)
for s in range(500):
    randoms.append(float(np.random.Generator(np.random.PCG64(s)).random()))
    normals.extend(
        float(x) for x in np.random.Generator(np.random.PCG64(s)).normal(size=3)
    )
    uniforms.append(float(np.random.Generator(np.random.PCG64(s)).uniform(lo, hi)))

print("rng.random()      :", h(randoms))
print("rng.normal(size=3):", h(normals))
print("rng.uniform(lo,hi):", h(uniforms))

with localcontext() as ctx:
    ctx.prec = 40
    dec = [float(Decimal(u).exp()) for u in randoms[:200]]
    dec += [float(Decimal(abs(u) + 0.5).ln()) for u in randoms[:200]]
    dec += [float((Decimal(u) * Decimal(u) + Decimal(3)).sqrt()) for u in randoms[:200]]
print("decimal exp/ln/sqrt:", h(dec))
print(
    "libm exp/log/hypot :",
    h(
        [math.exp(u) for u in randoms[:200]]
        + [math.log(abs(u) + 0.5) for u in randoms[:200]]
        + [math.hypot(u, u, 3.0) for u in randoms[:200]]
    ),
)

print("manifest digest    :", generate_release_manifest()["semantic_sha256"])
