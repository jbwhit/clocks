"""Recompute numeric claims from the story pages."""

import math

import numpy as np

from clocks import ClockArray, MassConfig, clock_rates

# --- one-clock-is-not-enough ---
near = MassConfig(positions=np.array([[2.0]]), masses=np.array([0.04]))
far = MassConfig(positions=np.array([[4.0]]), masses=np.array([0.08]))
one = ClockArray(positions=np.array([[0.0]]))
print("single clock:", clock_rates(near, one), clock_rates(far, one))
pos = np.array([[-6.0], [-3.0], [0.0], [3.0], [6.0]])
for h in (2.0,):
    arr = ClockArray(positions=pos, track_offset=h)
    d = np.abs(clock_rates(near, arr) - clock_rates(far, arr))
    print(
        f"h={h} |rate diff| per clock x={pos[:, 0]}: {np.array2string(d, precision=5)}"
    )

# --- earth surface, units page ---
GM_E, R_E, C2 = 3.986004418e14, 6.371e6, 8.987551787e16
print("2Phi/c^2 at Earth surface:", -2 * GM_E / (R_E * C2))
print("1-sqrt(0.9):", 1 - math.sqrt(0.9))
print("1 cm height:", 9.80665 * 0.01 / C2, " 1 mm:", 9.80665 * 1e-3 / C2)

# --- GPS ---
GM, R_GPS = 3.986004418e14, 26_561.75e3
day = 86400.0
grav = GM / C2 * (1 / R_E - 1 / R_GPS) * day * 1e6
v2 = GM / R_GPS
vel = -v2 / (2 * C2) * day * 1e6
print(f"GPS grav {grav:.2f} us/day, vel {vel:.2f}, net {grav + vel:.2f}")
print("net drift as range per hour (m):", (grav + vel) * 1e-6 / 24 * 299792458)

# --- echolocation coda ---
BASE = 0.1
GM_SUN, GM_JUP = 1.327124e20, 1.266865e17
AU = 1.495979e11
earth = 9.80665 * BASE / C2
sun_static = GM_SUN * BASE / (AU**2 * C2)
sun_tide = 2 * GM_SUN * R_E * BASE / (AU**3 * C2)
jup_tide = 2 * GM_JUP * R_E * BASE / ((4.2 * AU) ** 3 * C2)
jup_hover = GM_JUP * BASE / (1e8**2 * C2)
for name, s in [
    ("earth", earth),
    ("sun static", sun_static),
    ("sun tide", sun_tide),
    ("jup tide", jup_tide),
    ("jup hover", jup_hover),
]:
    tau = (1e-16 / s) ** 2
    print(
        f"{name:10s} {s:.3e}  tau={tau:.3e}s ({tau / 3.156e7:.3g} yr)  floor ratio {s / 1e-18:.3g}"
    )
horizon = math.sqrt(GM_SUN * BASE / (C2 * 1e-18))
print("horizon (AU):", horizon / AU, " AU/horizon:", AU / horizon)
print("baseline/AU:", BASE / AU)
print("Jupiter radius check: 1e8 - 71492e3 =", (1e8 - 71_492e3) / 1e3, "km")
# lunar tidal potential amplitude at surface (degree-2), fractional
GM_MOON, R_MOON = 4.9048695e12, 3.844e8
print("lunar tidal potential /c^2 ~", GM_MOON * R_E**2 / R_MOON**3 / C2)
print("solar tidal potential /c^2 ~", GM_SUN * R_E**2 / AU**3 / C2)
