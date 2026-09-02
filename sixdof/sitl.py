"""Software-in-the-loop harness.

This is the part that separates a SITL rig from a plain trajectory
integrator. Real flight software does not see the state vector. It sees
sensor measurements, sampled at a fixed rate, quantized by an ADC, corrupted
by noise and bias, and delivered one or more frames late. Then it computes a
command that does not reach the surfaces until the next actuator update.

The physics here runs at 500 Hz while the autopilot runs at 100 Hz, so four
physics steps pass between control updates and the command is held constant
across them, which is what a zero-order hold actually does. Skipping that
and calling the control law every derivative evaluation makes a marginally
stable loop look comfortable, because you have quietly given the controller
five times the bandwidth it will have on the vehicle.
"""

import math
import random

from .frames import euler_from_quat
from .dynamics import derivative, renormalize
from .integrate import rk4_step


class SensorSuite:
    """Rate gyro and attitude estimate with bias, noise, quantization, delay."""

    def __init__(self, seed=1, gyro_bias=None, gyro_noise=0.0015,
                 quantize=math.radians(0.01), delay_frames=1):
        self.rng = random.Random(seed)
        self.gyro_noise = gyro_noise
        self.quantize = quantize
        self.delay_frames = delay_frames
        self.gyro_bias = gyro_bias if gyro_bias is not None else (
            self.rng.gauss(0, 0.002), self.rng.gauss(0, 0.002), self.rng.gauss(0, 0.002))
        self.buffer = []

    def _q(self, value):
        if self.quantize <= 0:
            return value
        return round(value / self.quantize) * self.quantize

    def sample(self, state):
        roll, pitch, yaw = euler_from_quat((state[6], state[7], state[8], state[9]))
        p, q, r = state[10], state[11], state[12]
        meas = dict(
            roll=self._q(roll), pitch=self._q(pitch), yaw=self._q(yaw),
            p=self._q(p + self.gyro_bias[0] + self.rng.gauss(0, self.gyro_noise)),
            q=self._q(q + self.gyro_bias[1] + self.rng.gauss(0, self.gyro_noise)),
            r=self._q(r + self.gyro_bias[2] + self.rng.gauss(0, self.gyro_noise)),
            altitude=-state[2],
        )
        self.buffer.append(meas)
        # Deliver the frame from delay_frames ago; early on, the buffer is
        # short and the oldest available sample is returned instead.
        if len(self.buffer) > self.delay_frames:
            return self.buffer.pop(0)
        return self.buffer[0]


class PitchHold:
    """Discrete pitch-attitude hold with an inner rate loop.

    Sign convention matters more than the gains here. Elevator power is
    negative (cm_de < 0), so a positive deflection produces a nose-down
    moment. The command therefore has to be the negative of the usual PID
    expression: to raise the nose the surface goes negative. The first
    version of this class used the textbook positive form and drove the
    vehicle away from its target, diverging to about -50 degrees of pitch
    within 25 seconds. The test that catches it asserts convergence, not
    just boundedness, because a diverging loop still produces a finite
    trajectory that looks fine on a plot.

    Gains are otherwise conservative on purpose. The interesting result is
    not that this tracks well, it is that the same gains behave differently
    once the loop is closed at 100 Hz through delayed, quantized
    measurements instead of on perfect state.
    """

    def __init__(self, kp=2.6, kd=0.55, ki=0.35, limit=math.radians(20.0), dt=0.01):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.limit = limit
        self.dt = dt
        self.integral = 0.0
        self.target = 0.0

    def update(self, meas):
        error = self.target - meas["pitch"]
        self.integral += error * self.dt
        # Anti-windup: clamp the integrator to what the surface can deliver.
        windup_cap = self.limit / max(self.ki, 1e-6)
        self.integral = max(-windup_cap, min(windup_cap, self.integral))
        cmd = -(self.kp * error + self.ki * self.integral) + self.kd * meas["q"]
        return max(-self.limit, min(self.limit, cmd))


class AttitudeHold:
    """Pitch hold plus wings-level roll hold and a yaw damper.

    This exists because a pitch-only loop is not enough to fly the vehicle,
    and the Monte Carlo is what proved it. With only the elevator closed,
    a 2 m/s wind was enough to drive Euler pitch to -52 degrees. The pitch
    loop was not the problem: the wind produced a steady sideslip, the
    dihedral effect (cl_beta) rolled the airframe, nothing was holding the
    wings level, and once the vehicle banked past about 40 degrees the Euler
    pitch angle stopped meaning what the controller thought it meant. The
    dispersed cases were reporting a 30 degree mean pitch error that was
    really an uncontrolled roll-off.

    Lateral gains are deliberately simple: proportional on bank angle with
    rate damping, and a washout-free yaw damper on r. Enough to hold the
    wings level so that pitch tracking can actually be measured.
    """

    def __init__(self, dt=0.01, pitch=None,
                 kp_roll=1.9, kd_roll=0.42, aileron_limit=math.radians(20.0),
                 kr_yaw=0.55, rudder_limit=math.radians(15.0)):
        self.pitch = pitch or PitchHold(dt=dt)
        self.kp_roll = kp_roll
        self.kd_roll = kd_roll
        self.aileron_limit = aileron_limit
        self.kr_yaw = kr_yaw
        self.rudder_limit = rudder_limit

    @property
    def target(self):
        return self.pitch.target

    @target.setter
    def target(self, value):
        self.pitch.target = value

    def update(self, meas):
        elevator = self.pitch.update(meas)
        # cl_da is positive, so positive aileron rolls right. To level a
        # right-wing-down bank the command has to go negative.
        aileron = -(self.kp_roll * meas["roll"] + self.kd_roll * meas["p"])
        aileron = max(-self.aileron_limit, min(self.aileron_limit, aileron))
        # cn_dr is negative, so positive rudder yaws left; damp positive r
        # with positive rudder.
        rudder = self.kr_yaw * meas["r"]
        rudder = max(-self.rudder_limit, min(self.rudder_limit, rudder))
        return elevator, aileron, rudder


def run_sitl(state, vehicle, target_pitch_deg=4.0, physics_hz=500, control_hz=100,
             t_end=30.0, wind_ned=(0.0, 0.0, 0.0), thrust=0.0, seed=1,
             sensors=None, controller=None, sample_every=25, settle_time=None):
    """Closed-loop run with a zero-order hold between control updates.

    settle_time splits the error metrics into the whole run and the portion
    after the initial transient. Whole-run peak error is always just the
    initial step (the vehicle starts at zero pitch and is asked for five
    degrees), so it says nothing about tracking quality; the settled window
    is the number worth comparing across configurations.
    """
    if physics_hz % control_hz != 0:
        raise ValueError("physics rate must be an integer multiple of the control rate")
    decimation = physics_hz // control_hz
    dt = 1.0 / physics_hz

    sensors = sensors or SensorSuite(seed=seed)
    controller = controller or AttitudeHold(dt=1.0 / control_hz)
    controller.target = math.radians(target_pitch_deg)

    state = list(state)
    t = 0.0
    steps = 0
    commands = (0.0, 0.0, 0.0)
    samples = []
    control_updates = 0
    pitch_error_sum = 0.0
    pitch_error_peak = 0.0
    settle_from = 0.6 * t_end if settle_time is None else settle_time
    settled_sum = 0.0
    settled_peak = 0.0
    settled_count = 0
    diverged = False

    def deriv(tt, ss):
        return derivative(tt, ss, vehicle, commands, wind_ned, thrust)

    while t < t_end:
        if steps % decimation == 0:
            meas = sensors.sample(state)
            out = controller.update(meas)
            # A pitch-only controller returns a scalar; the full attitude
            # controller returns all three surfaces.
            commands = tuple(out) if isinstance(out, (tuple, list)) else (out, 0.0, 0.0)
            control_updates += 1
            err = abs(controller.target - meas["pitch"])
            pitch_error_sum += err
            pitch_error_peak = max(pitch_error_peak, err)
            if t >= settle_from:
                settled_sum += err
                settled_peak = max(settled_peak, err)
                settled_count += 1
            if err > math.radians(60.0):
                diverged = True

        if sample_every and steps % sample_every == 0:
            roll, pitch, yaw = euler_from_quat((state[6], state[7], state[8], state[9]))
            samples.append(dict(t=t, pitch=pitch, elevator=state[13], altitude=-state[2]))

        state = renormalize(rk4_step(t, state, dt, deriv))
        t += dt
        steps += 1
        if -state[2] <= 0.0:
            break

    roll, pitch, yaw = euler_from_quat((state[6], state[7], state[8], state[9]))
    return dict(
        t=t, state=state, samples=samples, steps=steps,
        control_updates=control_updates,
        physics_steps_per_control=decimation,
        final_pitch_deg=math.degrees(pitch),
        target_pitch_deg=target_pitch_deg,
        mean_abs_pitch_error_deg=math.degrees(pitch_error_sum / max(control_updates, 1)),
        peak_abs_pitch_error_deg=math.degrees(pitch_error_peak),
        settled_mean_abs_error_deg=math.degrees(settled_sum / max(settled_count, 1)),
        settled_peak_abs_error_deg=math.degrees(settled_peak),
        diverged=diverged,
    )

