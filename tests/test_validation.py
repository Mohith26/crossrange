"""Physics validation against closed-form answers.

These are the tests that decide whether the simulator is worth anything.
Each one sets up a case where the true answer is known analytically, then
checks the integrator reproduces it. A 6DOF sim that only agrees with itself
is a plotting tool.
"""

import math

from sixdof.frames import G0, atmosphere, quat_from_euler
from sixdof.vehicle import Vehicle
from sixdof.dynamics import make_state, energy
from sixdof.integrate import simulate, convergence_order


def no_aero_vehicle():
    """Reference area of zero removes every aerodynamic force and moment."""
    return Vehicle(s_ref=0.0)


def register(check, close):
    def test_vacuum_parabola():
        """Ballistic flight with no aero must match the closed-form parabola."""
        v0 = 250.0
        theta = math.radians(35.0)
        h0 = 3000.0
        vehicle = no_aero_vehicle()
        state = make_state(position=(0.0, 0.0, -h0), velocity=(v0, 0.0, 0.0),
                           quat=quat_from_euler(0.0, theta, 0.0))
        out = simulate(state, vehicle, dt=0.002, t_end=20.0, stop_on_ground=False)

        # Compare at the time the integrator actually reached. The loop exits
        # on t < t_end with an accumulating float, so the final time is within
        # one step of 20 s rather than exactly 20 s, and assuming otherwise
        # shows up as a bogus 0.4 m error at 250 m/s.
        t_fly = out["t"]
        vz_up = v0 * math.sin(theta)
        vx = v0 * math.cos(theta)
        expect_north = vx * t_fly
        expect_alt = h0 + vz_up * t_fly - 0.5 * G0 * t_fly * t_fly

        got_north = out["state"][0]
        got_alt = -out["state"][2]
        close("vacuum downrange matches closed form", got_north, expect_north, 1e-6)
        close("vacuum altitude matches closed form", got_alt, expect_alt, 1e-6)

    def test_vacuum_impact_time():
        """The ground-event bisection must find the analytic impact time."""
        v0 = 180.0
        theta = math.radians(20.0)
        h0 = 1500.0
        vehicle = no_aero_vehicle()
        state = make_state(position=(0.0, 0.0, -h0), velocity=(v0, 0.0, 0.0),
                           quat=quat_from_euler(0.0, theta, 0.0))
        out = simulate(state, vehicle, dt=0.01, t_end=200.0, stop_on_ground=True)

        vz_up = v0 * math.sin(theta)
        # h0 + vz t - g t^2 / 2 = 0
        disc = vz_up * vz_up + 2.0 * G0 * h0
        t_impact = (vz_up + math.sqrt(disc)) / G0
        close("impact time from bisection matches closed form", out["t"], t_impact, 1e-4)
        close("impact altitude is essentially zero", -out["state"][2], 0.0, 1e-3)
        expect_range = v0 * math.cos(theta) * t_impact
        close("impact downrange matches closed form", out["state"][0], expect_range, 0.05)

    def test_impact_range_independent_of_step():
        """Without event bisection this number moves with dt. With it, it should not."""
        vehicle = no_aero_vehicle()
        ranges = []
        for dt in (0.02, 0.01, 0.005):
            state = make_state(position=(0.0, 0.0, -1200.0), velocity=(200.0, 0.0, 0.0),
                               quat=quat_from_euler(0.0, math.radians(10.0), 0.0))
            out = simulate(state, vehicle, dt=dt, t_end=200.0, stop_on_ground=True)
            ranges.append(out["state"][0])
        spread = max(ranges) - min(ranges)
        check("impact range varies under 1 cm across a 4x step change (spread %.4f m)" % spread,
              spread < 0.01)

    def test_energy_conserved_without_drag():
        """No aero, no thrust: specific mechanical energy is constant."""
        vehicle = no_aero_vehicle()
        state = make_state(position=(0.0, 0.0, -6000.0), velocity=(220.0, 0.0, 0.0),
                           quat=quat_from_euler(0.0, math.radians(15.0), 0.0))
        e0 = energy(state, vehicle)
        out = simulate(state, vehicle, dt=0.002, t_end=30.0, stop_on_ground=False)
        e1 = energy(out["state"], vehicle)
        rel = abs(e1 - e0) / abs(e0)
        check("specific energy drifts under 1e-10 relative over 30 s (%.2e)" % rel, rel < 1e-10)

    def test_energy_decreases_with_drag():
        """Sanity in the other direction: with drag on, energy must fall."""
        vehicle = Vehicle()
        state = make_state(position=(0.0, 0.0, -6000.0), velocity=(220.0, 0.0, 0.0))
        e0 = energy(state, vehicle)
        out = simulate(state, vehicle, dt=0.002, t_end=20.0, stop_on_ground=False)
        e1 = energy(out["state"], vehicle)
        check("drag removes energy", e1 < e0)

    def _terminal_speed(vehicle, altitude):
        """Closed-form terminal speed, solved iteratively because CD0 is a
        function of Mach and Mach is a function of the answer."""
        from sixdof.vehicle import _interp1
        _, _, rho, a = atmosphere(altitude)
        v = 200.0
        for _ in range(60):
            cd0 = _interp1(vehicle.mach_grid, vehicle.cd0_grid, v / a)
            v_new = math.sqrt(2.0 * vehicle.mass * G0 / (rho * vehicle.s_ref * cd0))
            if abs(v_new - v) < 1e-9:
                v = v_new
                break
            v = 0.5 * (v + v_new)  # damped to keep it off the transonic cliff
        return v

    def test_terminal_velocity_force_balance():
        """At the terminal speed, drag must exactly cancel weight.

        This is the analytic half of the check and needs no integration.
        """
        vehicle = Vehicle(k_induced=0.0)
        altitude = 5000.0
        v_term = _terminal_speed(vehicle, altitude)
        from sixdof.frames import airdata
        air = airdata((v_term, 0.0, 0.0), altitude)
        coef = vehicle.coefficients(air, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        drag_force = air["qbar"] * vehicle.s_ref * coef["drag"]
        weight = vehicle.mass * G0
        rel = abs(drag_force - weight) / weight
        check("at terminal speed drag equals weight to 1e-6 relative (%.1f N vs %.1f N)"
              % (drag_force, weight), rel < 1e-6)

    def test_drag_ode_matches_tanh():
        """RK4 on the 1D drag equation must match the closed-form tanh.

        Vertical fall at constant density has an exact solution,
        v(t) = v_term * tanh(g t / v_term). Freezing density isolates the
        integrator and the drag model from the atmosphere gradient, so any
        disagreement here is a real error rather than the vehicle chasing a
        terminal speed that keeps moving.
        """
        vehicle = Vehicle(k_induced=0.0)
        altitude = 5000.0
        _, _, rho, a = atmosphere(altitude)
        from sixdof.vehicle import _interp1
        v_term = _terminal_speed(vehicle, altitude)
        cd0 = _interp1(vehicle.mach_grid, vehicle.cd0_grid, v_term / a)
        k = 0.5 * rho * vehicle.s_ref * cd0 / vehicle.mass

        def dv(t, s):
            return [G0 - k * s[0] * s[0]]

        from sixdof.integrate import rk4_step
        dt = 0.001
        s = [0.0]
        for i in range(20000):  # 20 seconds
            s = rk4_step(i * dt, s, dt, dv)
        t_final = 20000 * dt
        exact = v_term * math.tanh(G0 * t_final / v_term)
        rel = abs(s[0] - exact) / exact
        check("1D drag integration matches tanh solution to 1e-9 relative (%.2e)" % rel,
              rel < 1e-9)
        asymptote = math.sqrt(G0 / k)
        rel_asym = abs(v_term - asymptote) / asymptote
        check("the tanh asymptote is the terminal speed to 1e-9 relative (%.2e)" % rel_asym,
              rel_asym < 1e-9)

    def test_fall_approaches_terminal():
        """The full 6DOF drop must trend toward the local terminal speed.

        It cannot reach it: density rises as the vehicle descends, so the
        target keeps dropping and the trajectory lags. An earlier version of
        this asserted 2 percent and failed at 7.3 percent for exactly that
        reason, which is physics rather than a bug. What is checked is that
        the vehicle gets most of the way there and that axial acceleration
        has collapsed from its initial value of about one g.
        """
        vehicle = Vehicle(k_induced=0.0)
        state = make_state(position=(0.0, 0.0, -9500.0), velocity=(60.0, 0.0, 0.0),
                           quat=quat_from_euler(0.0, math.radians(-90.0), 0.0))
        out = simulate(state, vehicle, dt=0.002, t_end=60.0, stop_on_ground=True)
        st = out["state"]
        speed = math.sqrt(st[3] ** 2 + st[4] ** 2 + st[5] ** 2)
        v_term = _terminal_speed(vehicle, -st[2])
        frac = speed / v_term
        check("fall reaches at least 80 percent of local terminal speed (%.1f of %.1f m/s, %.0f%%)"
              % (speed, v_term, frac * 100), 0.80 <= frac <= 1.02)

        from sixdof.dynamics import derivative
        d = derivative(out["t"], st, vehicle, (0.0, 0.0, 0.0))
        accel = abs(d[3])
        check("axial acceleration has fallen below 0.6 g near terminal (%.2f m/s^2)" % accel,
              accel < 0.6 * G0)

    def smooth_aero_vehicle():
        """Constant coefficient tables, so the derivative is C-infinity.

        RK4's formal order only holds for smooth right-hand sides. The
        production tables are piecewise linear, so their derivative jumps at
        every Mach breakpoint the trajectory crosses.
        """
        n = 9
        return Vehicle(cd0_grid=[0.035] * n, cla_grid=[3.0] * n, cma_grid=[-0.65] * n)

    def test_rk4_convergence_order_smooth():
        """On a smooth right-hand side, step halving must show order ~4."""
        vehicle = smooth_aero_vehicle()
        state = make_state(position=(0.0, 0.0, -8000.0), velocity=(240.0, 5.0, -3.0),
                           quat=quat_from_euler(0.05, math.radians(6.0), 0.1),
                           rates=(0.02, -0.01, 0.015))
        res = convergence_order(state, vehicle, dt_coarse=0.04, t_end=8.0)
        check("convergence order on smooth aero is between 3.5 and 4.5 (%.2f)" % (res["order"] or -1),
              res["order"] is not None and 3.5 <= res["order"] <= 4.5)

    def test_production_tables_keep_convergence_order():
        """The piecewise-linear tables do not measurably cost order here.

        I expected them to. Linear interpolation is only C0, so the
        derivative jumps at every Mach breakpoint the trajectory crosses,
        and the formal RK4 order should not survive that. An earlier version
        of this test asserted the order would fall below 3 and it did appear
        to, reporting values between 0.3 and 1.1.

        That turned out to be an artifact of measuring per-axis. When one
        position component's coarse and mid solutions nearly coincide, the
        ratio is formed from two cancelling doubles and the estimate is
        meaningless. Switching to the norm of the position difference showed
        both the smooth and production tables converging at about 4. The
        breakpoints are simply crossed too rarely over 8 seconds to dominate
        the global error at these step sizes, so the honest assertion is
        that order is retained, not lost.
        """
        vehicle = Vehicle()
        state = make_state(position=(0.0, 0.0, -8000.0), velocity=(240.0, 5.0, -3.0),
                           quat=quat_from_euler(0.05, math.radians(6.0), 0.1),
                           rates=(0.02, -0.01, 0.015))
        res = convergence_order(state, vehicle, dt_coarse=0.04, t_end=8.0)
        check("production tables still converge between 3.5 and 4.5 (%.2f)" % (res["order"] or -1),
              res["order"] is not None and 3.5 <= res["order"] <= 4.5)

    def test_order_estimates_agree_across_axes():
        """Per-axis and norm-based order estimates should now agree.

        They did not for most of this project's life. The per-axis numbers
        swung between 0.3 and 6 while the norm sat near 4, and I attributed
        that to catastrophic cancellation in the per-axis ratios. That was
        wrong. The real cause was a variable shadowing bug in the rotational
        equations that fed the rudder actuator an angular acceleration, which
        made the lateral trajectory garbage and the error ratios meaningless
        with it. Once that was fixed the two estimates came into agreement,
        so this test asserts agreement rather than instability. The norm is
        still what the estimate reports, because it remains the more robust
        of the two, but the disagreement was never the real problem.
        """
        vehicle = Vehicle()
        state = make_state(position=(0.0, 0.0, -8000.0), velocity=(240.0, 5.0, -3.0),
                           quat=quat_from_euler(0.05, math.radians(6.0), 0.1),
                           rates=(0.02, -0.01, 0.015))
        res = convergence_order(state, vehicle, dt_coarse=0.02, t_end=8.0)
        spread = max(res["per_axis"]) - min(res["per_axis"]) if len(res["per_axis"]) > 1 else 0.0
        check("per-axis order estimates agree within 0.5 (spread %.2f, norm %.2f)"
              % (spread, res["order"] or -1), spread < 0.5)
        check("every per-axis estimate is itself near fourth order",
              all(3.0 <= o <= 5.0 for o in res["per_axis"]))

    def test_quaternion_norm_drift_bounded():
        """Measure how far RK4 pushes the quaternion off the unit sphere."""
        vehicle = Vehicle()
        state = make_state(position=(0.0, 0.0, -7000.0), velocity=(230.0, 0.0, 0.0),
                           rates=(0.9, 0.6, -0.4))
        out = simulate(state, vehicle, dt=0.002, t_end=20.0,
                       stop_on_ground=False, normalize=False)
        q = out["state"][6:10]
        n = math.sqrt(sum(c * c for c in q))
        drift = abs(n - 1.0)
        check("un-normalized quaternion norm drifts under 1e-6 over 20 s at 1 rad/s (%.2e)" % drift,
              drift < 1e-6)

    def test_pitch_stability_sign():
        """Static margin sanity: nose-up disturbance must produce nose-down moment."""
        vehicle = Vehicle()
        from sixdof.frames import airdata
        air = airdata((250.0, 0.0, 25.0), 5000.0)  # positive alpha
        coef = vehicle.coefficients(air, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        check("positive alpha produces a restoring (negative) pitch moment", coef["cm"] < 0)
        check("positive alpha produces positive lift", coef["lift"] > 0)

    def test_weathercock_sign():
        vehicle = Vehicle()
        from sixdof.frames import airdata
        air = airdata((250.0, 20.0, 0.0), 5000.0)  # positive beta
        coef = vehicle.coefficients(air, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        check("positive sideslip produces a restoring yaw moment", coef["cn"] > 0)
        check("positive sideslip produces a side force to the left", coef["cy"] < 0)

    def test_damping_opposes_rate():
        vehicle = Vehicle()
        from sixdof.frames import airdata
        air = airdata((250.0, 0.0, 0.0), 5000.0)
        pitch_up = vehicle.coefficients(air, (0.0, 0.5, 0.0), (0.0, 0.0, 0.0))
        check("pitch rate produces opposing pitch moment", pitch_up["cm"] < 0)
        roll = vehicle.coefficients(air, (0.5, 0.0, 0.0), (0.0, 0.0, 0.0))
        check("roll rate produces opposing roll moment", roll["cl"] < 0)
        yaw = vehicle.coefficients(air, (0.0, 0.0, 0.5), (0.0, 0.0, 0.0))
        check("yaw rate produces opposing yaw moment", yaw["cn"] < 0)

    def test_drag_never_negative_across_mach():
        """The reason the tables clamp instead of extrapolating."""
        vehicle = Vehicle()
        from sixdof.frames import airdata
        worst = 1.0
        ok = True
        for mach_target in [0.1, 0.5, 0.95, 1.1, 2.0, 4.0, 6.0, 9.0]:
            _, _, _, a = atmosphere(15000.0)
            air = airdata((mach_target * a, 0.0, 0.0), 15000.0)
            coef = vehicle.coefficients(air, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            if coef["drag"] <= 0:
                ok = False
            worst = min(worst, coef["drag"])
        check("drag stays positive from Mach 0.1 to 9 including past the table edge (min %.4f)" % worst, ok)

    def test_transonic_drag_rise():
        vehicle = Vehicle()
        from sixdof.vehicle import _interp1
        sub = _interp1(vehicle.mach_grid, vehicle.cd0_grid, 0.6)
        peak = _interp1(vehicle.mach_grid, vehicle.cd0_grid, 1.05)
        suporsonic = _interp1(vehicle.mach_grid, vehicle.cd0_grid, 3.0)
        check("drag rises through the transonic peak", peak > sub * 2.0)
        check("drag falls again supersonically", suporsonic < peak)

    def test_actuator_rate_limit():
        vehicle = Vehicle()
        act = vehicle.elevator
        # A huge step command cannot exceed the rate limit.
        rate = act.derivative(0.0, math.radians(1000.0))
        close("actuator rate is clamped to its limit", rate, act.rate_limit, 1e-12)
        # And the commanded position is clamped to the deflection limit.
        settled = act.derivative(act.limit, math.radians(90.0))
        close("at the deflection limit a further command produces no rate", settled, 0.0, 1e-12)

    def test_actuator_first_order_response():
        """A first-order lag reaches 63.2 percent of a step in one time constant."""
        vehicle = Vehicle()
        act = vehicle.elevator
        target = math.radians(5.0)  # small enough not to hit the rate limit
        state = 0.0
        dt = 0.0005
        steps = int(act.tau / dt)
        for _ in range(steps):
            state += act.derivative(state, target) * dt
        frac = state / target
        close("step response reaches 1 - 1/e after one time constant", frac, 0.6321, 0.01)

    return [test_vacuum_parabola, test_vacuum_impact_time, test_impact_range_independent_of_step,
            test_energy_conserved_without_drag, test_energy_decreases_with_drag,
            test_terminal_velocity_force_balance, test_drag_ode_matches_tanh,
            test_fall_approaches_terminal,
            test_rk4_convergence_order_smooth, test_production_tables_keep_convergence_order,
            test_order_estimates_agree_across_axes,
            test_quaternion_norm_drift_bounded, test_pitch_stability_sign,
            test_weathercock_sign, test_damping_opposes_rate,
            test_drag_never_negative_across_mach, test_transonic_drag_rise,
            test_actuator_rate_limit, test_actuator_first_order_response]

