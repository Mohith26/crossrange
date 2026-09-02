"""Fixed-step RK4 with ground-impact event detection.

Fixed step rather than adaptive on purpose. A software-in-the-loop rig has
to hold a deterministic relationship between the physics step and the flight
software step, and an adaptive integrator that shortens its step during a
transient quietly changes how many physics steps fall between two autopilot
updates. Monte Carlo repeatability goes with it.

Ground impact is found by bisecting the last step rather than accepting
whatever altitude the step happened to land on. Without it, reported impact
range depends on step size, which makes two Monte Carlo sets with different
dt incomparable.
"""

from .dynamics import derivative, renormalize


def rk4_step(t, state, dt, deriv_fn):
    k1 = deriv_fn(t, state)
    s2 = [state[i] + 0.5 * dt * k1[i] for i in range(len(state))]
    k2 = deriv_fn(t + 0.5 * dt, s2)
    s3 = [state[i] + 0.5 * dt * k2[i] for i in range(len(state))]
    k3 = deriv_fn(t + 0.5 * dt, s3)
    s4 = [state[i] + dt * k3[i] for i in range(len(state))]
    k4 = deriv_fn(t + dt, s4)
    return [state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            for i in range(len(state))]


def simulate(state, vehicle, controller=None, dt=0.002, t_end=60.0,
             wind_ned=(0.0, 0.0, 0.0), thrust=0.0, sample_every=0,
             stop_on_ground=True, normalize=True):
    """Run the vehicle forward. Returns a result dict.

    controller, if given, is called as controller(t, state) and returns the
    three surface commands. It is called once per physics step here; the
    SITL harness in sitl.py is what introduces a realistic slower rate.
    """
    state = list(state)
    samples = []
    commands = (0.0, 0.0, 0.0)
    impact = None

    def deriv(tt, ss):
        return derivative(tt, ss, vehicle, commands, wind_ned, thrust)

    # The step count is fixed up front and time is derived as step * dt rather
    # than accumulated by repeated addition. Accumulating drifts by a few ulp
    # per step, which is harmless for a plot but means two runs at different
    # dt stop at slightly different final times. That made the step-halving
    # convergence estimate report order 0.33 instead of 4, because it was
    # differencing states taken at different instants.
    n_steps = int(round(t_end / dt))
    steps = 0
    t = 0.0

    for i in range(n_steps):
        t = i * dt
        if controller is not None:
            commands = controller(t, state)
        if sample_every and i % sample_every == 0:
            samples.append((t, list(state)))

        prev_state = list(state)
        state = rk4_step(t, state, dt, deriv)
        if normalize:
            state = renormalize(state)
        steps = i + 1
        t = steps * dt

        if stop_on_ground and -state[2] <= 0.0 and -prev_state[2] > 0.0:
            impact = _bisect_ground(i * dt, prev_state, dt, deriv, normalize)
            state = impact["state"]
            t = impact["t"]
            break

    samples.append((t, list(state)))
    return dict(t=t, state=state, samples=samples, steps=steps, impact=impact)


def _bisect_ground(t0, state0, dt, deriv, normalize, iterations=40):
    """Find the sub-step time where altitude crosses zero."""
    lo, hi = 0.0, dt
    best = None
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        trial = rk4_step(t0, state0, mid, deriv)
        if normalize:
            trial = renormalize(trial)
        altitude = -trial[2]
        best = dict(t=t0 + mid, state=trial, altitude=altitude)
        if altitude > 0.0:
            lo = mid
        else:
            hi = mid
    return best


def convergence_order(state, vehicle, dt_coarse, t_end, wind_ned=(0.0, 0.0, 0.0)):
    """Estimate the observed order of the integrator by step halving.

    Runs the same trajectory at dt, dt/2 and dt/4 with the ground event and
    quaternion renormalization disabled (both are non-smooth and would
    contaminate the estimate), then applies the standard three-grid formula.
    """
    def run(dt):
        return simulate(state, vehicle, dt=dt, t_end=t_end, wind_ned=wind_ned,
                        stop_on_ground=False, normalize=False)["state"]

    a = run(dt_coarse)
    b = run(dt_coarse / 2.0)
    c = run(dt_coarse / 4.0)

    import math

    def norm_diff(x, y):
        return math.sqrt(sum((x[i] - y[i]) ** 2 for i in range(3)))

    # The order is taken from the norm of the position difference, not from
    # each axis separately. Per-axis estimates are unstable: when one
    # component's coarse and mid solutions happen to nearly coincide, the
    # ratio is formed from two nearly-cancelling doubles and the reported
    # order swings between 1 and 6 with no physical meaning. The norm is
    # dominated by the component that actually carries the error.
    num = norm_diff(a, b)
    den = norm_diff(b, c)
    order = math.log(num / den) / math.log(2.0) if (num > 1e-12 and den > 1e-12) else None

    per_axis = []
    for i in range(3):
        n_i = abs(a[i] - b[i])
        d_i = abs(b[i] - c[i])
        if d_i > 1e-12 and n_i > 1e-12:
            per_axis.append(math.log(n_i / d_i) / math.log(2.0))

    return dict(order=order, per_axis=per_axis, mean=order,
                coarse_norm=num, fine_norm=den, coarse=a, mid=b, fine=c)

