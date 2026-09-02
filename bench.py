"""Produces bench-results.json. Every number quoted anywhere comes from here.

Timings are wall clock in CPython with no numpy: the state is 16 doubles and
the derivative is scalar arithmetic, so a vectorised rewrite would be faster
but would also hide where the cost actually is. The throughput figure that
matters for this kind of tool is Monte Carlo cases per minute, not raw steps
per second.
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sixdof.frames import G0, atmosphere, quat_from_euler
from sixdof.vehicle import Vehicle, _interp1
from sixdof.dynamics import make_state, energy
from sixdof.integrate import simulate, convergence_order
from sixdof.sitl import run_sitl
from sixdof.montecarlo import campaign


def bench_step_rate():
    vehicle = Vehicle()
    state = make_state(position=(0.0, 0.0, -9000.0), velocity=(250.0, 3.0, -2.0),
                       rates=(0.05, 0.02, -0.03))
    dt = 0.002
    t_end = 20.0
    t0 = time.time()
    out = simulate(state, vehicle, dt=dt, t_end=t_end, stop_on_ground=False)
    elapsed = time.time() - t0
    return dict(
        steps=out["steps"], seconds=round(elapsed, 3),
        steps_per_second=int(out["steps"] / elapsed),
        derivative_evals=out["steps"] * 4,
        derivative_evals_per_second=int(out["steps"] * 4 / elapsed),
        simulated_seconds_per_wall_second=round(t_end / elapsed, 1),
    )


def bench_vacuum_accuracy():
    v0, theta, h0 = 250.0, math.radians(35.0), 3000.0
    vehicle = Vehicle(s_ref=0.0)
    state = make_state(position=(0.0, 0.0, -h0), velocity=(v0, 0.0, 0.0),
                       quat=quat_from_euler(0.0, theta, 0.0))
    out = simulate(state, vehicle, dt=0.002, t_end=20.0, stop_on_ground=False)
    t = out["t"]
    expect_north = v0 * math.cos(theta) * t
    expect_alt = h0 + v0 * math.sin(theta) * t - 0.5 * G0 * t * t
    return dict(
        flight_time=t,
        downrange_error_m=abs(out["state"][0] - expect_north),
        altitude_error_m=abs(-out["state"][2] - expect_alt),
    )


def bench_energy_drift():
    vehicle = Vehicle(s_ref=0.0)
    state = make_state(position=(0.0, 0.0, -6000.0), velocity=(220.0, 0.0, 0.0),
                       quat=quat_from_euler(0.0, math.radians(15.0), 0.0))
    e0 = energy(state, vehicle)
    out = simulate(state, vehicle, dt=0.002, t_end=30.0, stop_on_ground=False)
    e1 = energy(out["state"], vehicle)
    return dict(relative_drift=abs(e1 - e0) / abs(e0), seconds_simulated=30.0)


def bench_quaternion_drift():
    vehicle = Vehicle()
    state = make_state(position=(0.0, 0.0, -7000.0), velocity=(230.0, 0.0, 0.0),
                       rates=(0.9, 0.6, -0.4))
    out = simulate(state, vehicle, dt=0.002, t_end=20.0, stop_on_ground=False, normalize=False)
    q = out["state"][6:10]
    return dict(norm_error_without_renormalization=abs(math.sqrt(sum(c * c for c in q)) - 1.0),
                seconds_simulated=20.0, body_rate_magnitude_rad_s=math.sqrt(0.9 ** 2 + 0.6 ** 2 + 0.4 ** 2))


def bench_convergence():
    smooth = Vehicle(cd0_grid=[0.035] * 9, cla_grid=[3.0] * 9, cma_grid=[-0.65] * 9)
    prod = Vehicle()
    state = make_state(position=(0.0, 0.0, -8000.0), velocity=(240.0, 5.0, -3.0),
                       quat=quat_from_euler(0.05, math.radians(6.0), 0.1),
                       rates=(0.02, -0.01, 0.015))
    a = convergence_order(state, smooth, dt_coarse=0.04, t_end=8.0)
    b = convergence_order(state, prod, dt_coarse=0.04, t_end=8.0)
    c = convergence_order(state, prod, dt_coarse=0.02, t_end=8.0)
    return dict(
        smooth_tables_order=round(a["order"], 3),
        production_tables_order=round(b["order"], 3),
        per_axis_spread_at_dt_0p02=round(max(c["per_axis"]) - min(c["per_axis"]), 3),
        norm_order_at_dt_0p02=round(c["order"], 3),
    )


def bench_impact_step_independence():
    vehicle = Vehicle(s_ref=0.0)
    ranges = {}
    for dt in (0.02, 0.01, 0.005):
        state = make_state(position=(0.0, 0.0, -1200.0), velocity=(200.0, 0.0, 0.0),
                           quat=quat_from_euler(0.0, math.radians(10.0), 0.0))
        out = simulate(state, vehicle, dt=dt, t_end=200.0, stop_on_ground=True)
        ranges["dt_%g" % dt] = out["state"][0]
    vals = list(ranges.values())
    return dict(ranges_m=ranges, spread_m=max(vals) - min(vals))


def bench_terminal():
    vehicle = Vehicle(k_induced=0.0)
    altitude = 5000.0
    _, _, rho, a = atmosphere(altitude)
    v = 200.0
    for _ in range(80):
        cd0 = _interp1(vehicle.mach_grid, vehicle.cd0_grid, v / a)
        v = 0.5 * (v + math.sqrt(2.0 * vehicle.mass * G0 / (rho * vehicle.s_ref * cd0)))
    k = 0.5 * rho * vehicle.s_ref * _interp1(vehicle.mach_grid, vehicle.cd0_grid, v / a) / vehicle.mass

    def dv(t, s):
        return [G0 - k * s[0] * s[0]]

    from sixdof.integrate import rk4_step
    dt = 0.001
    s = [0.0]
    for i in range(20000):
        s = rk4_step(i * dt, s, dt, dv)
    exact = v * math.tanh(G0 * 20.0 / v)
    return dict(terminal_speed_m_s=round(v, 3),
                tanh_relative_error=abs(s[0] - exact) / exact)


def bench_sitl():
    """Ideal state feedback versus a realistic sensor path, then a rate sweep.

    The headline result is not what I expected. At 100 Hz with a one-frame
    delay, gyro bias, noise and 0.01 degree quantization, the loop tracks
    essentially as well as perfect full-rate state feedback. The gains are
    conservative enough to absorb it. The sweep is there to find the rate
    where that stops being true, which is the number actually worth knowing
    before picking a flight-software schedule.
    """
    vehicle = Vehicle()
    state = make_state(position=(0.0, 0.0, -5000.0), velocity=(200.0, 0.0, 0.0))
    out_ideal = run_sitl(state, vehicle, target_pitch_deg=5.0, physics_hz=500, control_hz=500,
                         t_end=25.0, sample_every=0, sensors=_perfect_sensors())
    out_real = run_sitl(state, vehicle, target_pitch_deg=5.0, physics_hz=500, control_hz=100,
                        t_end=25.0, sample_every=0, seed=11)

    sweep = []
    for hz in (500, 250, 100, 50, 25, 10):
        o = run_sitl(state, vehicle, target_pitch_deg=5.0, physics_hz=500, control_hz=hz,
                     t_end=25.0, sample_every=0, seed=11)
        sweep.append(dict(control_hz=hz,
                          settled_mean_error_deg=round(o["settled_mean_abs_error_deg"], 4),
                          settled_peak_error_deg=round(o["settled_peak_abs_error_deg"], 3),
                          final_pitch_deg=round(o["final_pitch_deg"], 3),
                          diverged=o["diverged"]))

    return dict(
        ideal_full_rate=dict(final_pitch_deg=round(out_ideal["final_pitch_deg"], 3),
                             settled_mean_error_deg=round(out_ideal["settled_mean_abs_error_deg"], 4),
                             control_updates=out_ideal["control_updates"]),
        realistic_100hz_delayed=dict(final_pitch_deg=round(out_real["final_pitch_deg"], 3),
                                     settled_mean_error_deg=round(out_real["settled_mean_abs_error_deg"], 4),
                                     control_updates=out_real["control_updates"],
                                     physics_steps_per_control=out_real["physics_steps_per_control"]),
        control_rate_sweep=sweep,
    )


def _perfect_sensors():
    from sixdof.sitl import SensorSuite
    return SensorSuite(seed=1, gyro_bias=(0.0, 0.0, 0.0), gyro_noise=0.0,
                       quantize=0.0, delay_frames=0)


def bench_montecarlo(cases=200):
    t0 = time.time()
    out = campaign(cases=cases, t_end=20.0)
    elapsed = time.time() - t0
    return dict(
        cases=out["cases"], completed=out["completed"], failures=len(out["failures"]),
        seconds=round(elapsed, 2),
        cases_per_minute=round(out["completed"] / (elapsed / 60.0), 1),
        downrange_mean_m=round(out["downrange"]["mean"], 1),
        downrange_std_m=round(out["downrange"]["std"], 1),
        crossrange_std_m=round(out["crossrange"]["std"], 2),
        crossrange_p05_m=round(out["crossrange"]["p05"], 2),
        crossrange_p95_m=round(out["crossrange"]["p95"], 2),
        mean_pitch_error_deg=round(out["mean_pitch_error_deg"]["mean"], 3),
        peak_pitch_error_deg_p95=round(out["peak_pitch_error_deg"]["p95"], 3),
    )


def main():
    results = dict(
        note="pure-CPython, no numpy; timings from a single-threaded run",
        step_rate=bench_step_rate(),
        vacuum_accuracy=bench_vacuum_accuracy(),
        energy_drift=bench_energy_drift(),
        quaternion_drift=bench_quaternion_drift(),
        convergence=bench_convergence(),
        impact_step_independence=bench_impact_step_independence(),
        terminal=bench_terminal(),
        sitl=bench_sitl(),
        montecarlo=bench_montecarlo(),
    )
    print(json.dumps(results, indent=2))
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "bench-results.json"), "w") as fh:
        fh.write(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    main()

