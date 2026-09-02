"""Monte Carlo dispersion.

Every case is driven by its own seeded generator derived from a single
campaign seed, so case 47 is reproducible without replaying the 46 before
it. That sounds pedantic until a case diverges and you want to re-run only
that one under a debugger.

Dispersed parameters are the ones that actually move an impact point:
mass, the two aero coefficients the trajectory is most sensitive to, and
the wind. Everything else is held nominal so the scatter can be attributed.
"""

import math
import random

from .vehicle import Vehicle
from .dynamics import make_state
from .sitl import run_sitl, SensorSuite


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = fraction * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (idx - lo) * (sorted_values[hi] - sorted_values[lo])


def summarize(values):
    if not values:
        return None
    vs = sorted(values)
    n = len(vs)
    mean = sum(vs) / n
    var = sum((v - mean) ** 2 for v in vs) / (n - 1) if n > 1 else 0.0
    return dict(
        n=n, mean=mean, std=math.sqrt(var), min=vs[0], max=vs[-1],
        p05=percentile(vs, 0.05), p50=percentile(vs, 0.50), p95=percentile(vs, 0.95),
    )


def draw_case(case_seed, dispersions):
    rng = random.Random(case_seed)
    d = {}
    for name, (nominal, sigma) in dispersions.items():
        d[name] = nominal * (1.0 + rng.gauss(0.0, sigma)) if nominal != 0 else rng.gauss(0.0, sigma)
    wind_speed = abs(rng.gauss(0.0, 6.0))
    wind_dir = rng.uniform(0.0, 2.0 * math.pi)
    d["_wind"] = (wind_speed * math.cos(wind_dir), wind_speed * math.sin(wind_dir), 0.0)
    d["_sensor_seed"] = rng.randint(1, 10 ** 6)
    return d


DEFAULT_DISPERSIONS = {
    "mass": (1200.0, 0.03),        # 3 percent, 1 sigma
    "k_induced": (0.13, 0.10),     # 10 percent on the drag polar
    "cm_de": (-0.92, 0.08),        # 8 percent on elevator power
    "iyy": (8500.0, 0.05),
}


def campaign(cases=200, campaign_seed=20260901, t_end=25.0,
             dispersions=None, target_pitch_deg=4.0, altitude=4000.0, speed=180.0,
             start=0):
    """Run cases [start, start + cases).

    The start offset exists so a long campaign can be split across several
    processes or calls and reassembled without changing any case's seed.
    Case 137 draws the same dispersions whether it ran in the first chunk or
    the fourth.
    """
    dispersions = dispersions or DEFAULT_DISPERSIONS
    results = []
    failures = []

    for i in range(start, start + cases):
        case_seed = campaign_seed * 1000 + i
        draw = draw_case(case_seed, dispersions)
        params = {k: v for k, v in draw.items() if not k.startswith("_")}
        try:
            vehicle = Vehicle(**params)
            state = make_state(position=(0.0, 0.0, -altitude), velocity=(speed, 0.0, 0.0))
            out = run_sitl(state, vehicle, target_pitch_deg=target_pitch_deg,
                           t_end=t_end, wind_ned=draw["_wind"],
                           seed=draw["_sensor_seed"], sample_every=0)
            results.append(dict(
                case=i,
                downrange=out["state"][0],
                crossrange=out["state"][1],
                altitude=-out["state"][2],
                final_pitch_deg=out["final_pitch_deg"],
                mean_pitch_error_deg=out["mean_abs_pitch_error_deg"],
                peak_pitch_error_deg=out["peak_abs_pitch_error_deg"],
                flight_time=out["t"],
            ))
        except Exception as exc:  # a diverged case should be counted, not fatal
            failures.append(dict(case=i, seed=case_seed, error=repr(exc)))

    return dict(
        cases=cases,
        completed=len(results),
        failures=failures,
        downrange=summarize([r["downrange"] for r in results]),
        crossrange=summarize([r["crossrange"] for r in results]),
        mean_pitch_error_deg=summarize([r["mean_pitch_error_deg"] for r in results]),
        peak_pitch_error_deg=summarize([r["peak_pitch_error_deg"] for r in results]),
        flight_time=summarize([r["flight_time"] for r in results]),
        raw=results,
    )

