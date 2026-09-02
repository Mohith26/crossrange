"""Software-in-the-loop behaviour and Monte Carlo bookkeeping."""

import math

from sixdof.vehicle import Vehicle
from sixdof.dynamics import make_state
from sixdof.frames import quat_from_euler
from sixdof.sitl import run_sitl, SensorSuite, PitchHold
from sixdof.montecarlo import campaign, draw_case, summarize, DEFAULT_DISPERSIONS


def register(check, close):
    def test_rate_decimation():
        vehicle = Vehicle()
        state = make_state(position=(0.0, 0.0, -4000.0), velocity=(180.0, 0.0, 0.0))
        out = run_sitl(state, vehicle, physics_hz=500, control_hz=100, t_end=2.0, sample_every=0)
        close("five physics steps per control update", out["physics_steps_per_control"], 5, 0)
        expected_updates = out["steps"] // 5 + 1
        check("control update count matches the decimation (%d updates over %d steps)"
              % (out["control_updates"], out["steps"]),
              abs(out["control_updates"] - expected_updates) <= 1)

    def test_non_integer_rate_rejected():
        vehicle = Vehicle()
        state = make_state()
        raised = False
        try:
            run_sitl(state, vehicle, physics_hz=500, control_hz=300, t_end=0.5)
        except ValueError:
            raised = True
        check("a control rate that does not divide the physics rate is rejected", raised)

    def test_pitch_hold_converges():
        vehicle = Vehicle()
        state = make_state(position=(0.0, 0.0, -5000.0), velocity=(200.0, 0.0, 0.0))
        out = run_sitl(state, vehicle, target_pitch_deg=5.0, t_end=25.0, sample_every=0)
        err = abs(out["final_pitch_deg"] - 5.0)
        check("pitch hold settles within 0.5 deg of a 5 deg target (final %.2f deg)"
              % out["final_pitch_deg"], err < 0.5)

    def test_sensor_delay_is_real():
        """The delayed frame must actually differ from the instantaneous one."""
        sensors = SensorSuite(seed=7, gyro_noise=0.0, quantize=0.0, delay_frames=2)
        states = []
        for i in range(5):
            st = make_state(rates=(0.0, 0.1 * i, 0.0))
            states.append(sensors.sample(st))
        check("a two-frame delay returns an older measurement",
              abs(states[-1]["q"] - 0.4) > 1e-9)

    def test_quantization_bins():
        # Bias has to be zeroed explicitly; the default draws a random bias,
        # which is what a real IMU does and what made the first version of
        # this test read 3 degrees where 2 was expected.
        sensors = SensorSuite(seed=3, gyro_bias=(0.0, 0.0, 0.0), gyro_noise=0.0,
                              quantize=math.radians(1.0), delay_frames=0)
        st = make_state(rates=(0.0, math.radians(2.4), 0.0))
        meas = sensors.sample(st)
        # 2.4 degrees quantized to 1 degree bins lands on 2 degrees.
        close("gyro quantizes to the nearest bin", math.degrees(meas["q"]), 2.0, 1e-9)

    def test_zero_noise_is_deterministic():
        vehicle = Vehicle()
        state = make_state(position=(0.0, 0.0, -4000.0), velocity=(190.0, 0.0, 0.0))
        a = run_sitl(state, vehicle, t_end=6.0, seed=42, sample_every=0)
        b = run_sitl(state, vehicle, t_end=6.0, seed=42, sample_every=0)
        same = all(abs(a["state"][i] - b["state"][i]) < 1e-15 for i in range(16))
        check("the same seed reproduces the run exactly", same)

    def test_different_seed_differs():
        vehicle = Vehicle()
        state = make_state(position=(0.0, 0.0, -4000.0), velocity=(190.0, 0.0, 0.0))
        a = run_sitl(state, vehicle, t_end=6.0, seed=1, sample_every=0)
        b = run_sitl(state, vehicle, t_end=6.0, seed=2, sample_every=0)
        check("a different sensor seed changes the trajectory",
              abs(a["state"][0] - b["state"][0]) > 1e-9)

    def test_anti_windup_clamps():
        ctrl = PitchHold(dt=0.01)
        ctrl.target = math.radians(45.0)  # unreachable, will saturate
        meas = dict(pitch=0.0, q=0.0)
        for _ in range(2000):
            ctrl.update(meas)
        cap = ctrl.limit / ctrl.ki
        check("integrator is clamped rather than winding up without bound (%.2f vs cap %.2f)"
              % (ctrl.integral, cap), abs(ctrl.integral) <= cap + 1e-9)

    def test_wind_moves_the_trajectory():
        vehicle = Vehicle()
        state = make_state(position=(0.0, 0.0, -5000.0), velocity=(200.0, 0.0, 0.0))
        calm = run_sitl(state, vehicle, t_end=12.0, sample_every=0)
        gust = run_sitl(state, vehicle, t_end=12.0, wind_ned=(0.0, 25.0, 0.0), sample_every=0)
        drift = abs(gust["state"][1] - calm["state"][1])
        check("a 25 m/s crosswind moves crossrange by more than a metre (%.1f m)" % drift, drift > 1.0)

    def test_draw_case_reproducible():
        a = draw_case(12345, DEFAULT_DISPERSIONS)
        b = draw_case(12345, DEFAULT_DISPERSIONS)
        check("a case seed reproduces its own draw", a["mass"] == b["mass"] and a["_wind"] == b["_wind"])
        c = draw_case(12346, DEFAULT_DISPERSIONS)
        check("a different case seed draws differently", a["mass"] != c["mass"])

    def test_summarize_math():
        s = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
        close("mean", s["mean"], 3.0, 1e-12)
        close("median", s["p50"], 3.0, 1e-12)
        close("sample standard deviation", s["std"], math.sqrt(2.5), 1e-12)
        close("min", s["min"], 1.0, 1e-12)
        close("max", s["max"], 5.0, 1e-12)

    def test_small_campaign():
        out = campaign(cases=12, t_end=8.0)
        check("every case completed (%d of %d)" % (out["completed"], out["cases"]),
              out["completed"] == out["cases"])
        check("no diverged cases", not out["failures"])
        check("dispersions produce real scatter in crossrange (std %.2f m)"
              % out["crossrange"]["std"], out["crossrange"]["std"] > 0.0)

    def test_campaign_reproducible():
        a = campaign(cases=6, t_end=5.0, campaign_seed=999)
        b = campaign(cases=6, t_end=5.0, campaign_seed=999)
        same = abs(a["downrange"]["mean"] - b["downrange"]["mean"]) < 1e-12
        check("the same campaign seed reproduces the same statistics", same)

    return [test_rate_decimation, test_non_integer_rate_rejected, test_pitch_hold_converges,
            test_sensor_delay_is_real, test_quantization_bins, test_zero_noise_is_deterministic,
            test_different_seed_differs, test_anti_windup_clamps, test_wind_moves_the_trajectory,
            test_draw_case_reproducible, test_summarize_math, test_small_campaign,
            test_campaign_reproducible]

