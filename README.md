# crossrange

A 6DOF flight dynamics simulator with a software-in-the-loop harness and a
Monte Carlo dispersion campaign. Pure Python, standard library only, no
numpy.

I wrote this because "6DOF simulation" is easy to say and hard to check. The
equations of motion are in every textbook; what is not in the textbook is
how you convince yourself the thing you typed is right. Most of the effort
here went into the tests, not the physics.

## What it does

- Quaternion attitude propagation, so the vehicle can pitch through vertical
  without gimbal lock. Euler angles are derived on the way out for logging.
- Body-axis velocity with the transport term, full moment equations
  including the Ixz product of inertia (roll and yaw solved together).
- 1976 standard atmosphere through 51 km, coefficient build-up interpolated
  on Mach and angle of attack, first-order actuators with rate and position
  limits.
- Fixed-step RK4 with ground-impact bisection, so reported impact range does
  not move when you change the step size.
- A SITL harness where the physics runs at 500 Hz and flight software runs
  at 100 Hz through delayed, biased, quantized sensor measurements with a
  zero-order hold on the surfaces.
- Seeded Monte Carlo where every case is independently reproducible from its
  own derived seed.

## Validation

85 checks, all passing (`python3 tests/run.py`). The ones that matter are
the ones with a closed-form answer to compare against:

| Check | Result |
| --- | --- |
| Ballistic trajectory vs analytic parabola (20 s, no aero) | 7.5e-10 m downrange error |
| Ground impact time vs closed form | matches to 1e-4 s |
| Impact range across a 4x step change (0.02 / 0.01 / 0.005 s) | 3.5e-10 m spread |
| Specific energy drift, no drag, 30 s | 2.6e-13 relative |
| 1D drag ODE vs `v_term * tanh(g t / v_term)` | 1.4e-15 relative |
| Drag equals weight at terminal speed | exact to 1e-6 relative |
| RK4 observed convergence order (step halving, norm-based) | 4.05 |
| Quaternion norm drift over 20 s at 1.15 rad/s, un-normalized | 6.7e-15 |
| ISA temperature, pressure, density, speed of sound at 0 / 11 / 20 / 32 km | matches tabulated values |

Plus sign conventions that are easy to get backwards and produce plausible
garbage: positive alpha must give a restoring nose-down moment, positive
sideslip must weathercock, every body rate must be opposed by its damping
derivative, and drag must stay positive from Mach 0.1 to 9 including past
the edge of the table.

## Measured performance

From `bench.py`, CPython 3.12:

- 10,020 integration steps/sec, 40,080 derivative evaluations/sec, about
  20x real time at a 2 ms step.
- Control rate sweep, 25 s pitch hold, settled mean absolute error:

| Control rate | Settled error |
| --- | --- |
| 500 Hz | 0.228 deg |
| 250 Hz | 0.229 deg |
| 100 Hz | 0.229 deg |
| 50 Hz | 0.229 deg |
| 25 Hz | 0.228 deg |
| 10 Hz | 5.180 deg |

  The loop is flat from 500 Hz down to 25 Hz and then falls apart between
  25 and 10 Hz. That is the number worth knowing before picking a flight
  software schedule, and it is not the result I expected: I assumed the
  realistic sensor path (one-frame delay, gyro bias, noise, 0.01 degree
  quantization) would cost something measurable at 100 Hz against perfect
  full-rate state feedback. It costs 0.001 degrees.

- Monte Carlo, 100 cases, 10 s each, dispersing mass (3% 1-sigma), induced
  drag (10%), elevator power (8%), pitch inertia (5%) and wind (half-normal,
  sigma 6 m/s, random heading): downrange 1783.8 m with 3.0 m standard
  deviation, crossrange -0.13 m with 1.55 m standard deviation, mean pitch
  tracking error 1.21 degrees. Zero diverged cases.

## Three bugs worth writing down

**Variable shadowing in the rotational equations.** I named the yaw
acceleration `dr`, which shadowed the rudder deflection `dr` unpacked from
the state vector twelve lines earlier. The rudder actuator was being handed
an angular acceleration as its current position. A 5 degree aileron step
produced 204 deg/s of yaw rate and rolled the vehicle the wrong way.

This one is worth dwelling on because of how it hid. With a pitch-only
controller and no wind, the yaw rate stays near zero, the corrupted value
stays near zero, and every test passed. It only appeared once something
excited the lateral axis. I found it by chasing a Monte Carlo that reported
a 30 degree mean pitch error, assumed the controller was fragile, added a
wings-level roll hold, watched that diverge too, and only then ran an
open-loop aileron step and saw the vehicle roll the wrong way. Fixing it
moved the campaign's crossrange scatter from 126 m standard deviation to
1.55 m.

**Accumulated time in the integrator.** The loop advanced time with
`t += dt`. Over thousands of steps that drifts by a few ulp, so two runs at
different step sizes finished at slightly different final times. The
step-halving convergence estimate was differencing states taken at
different instants and reported order 0.33 instead of 4. Time is now
derived as `step * dt`.

**A wrong diagnosis I want to keep visible.** When the per-axis convergence
estimates swung between 0.3 and 6 while the norm-based estimate sat near 4,
I concluded the per-axis metric was numerically unstable and wrote a test
asserting it. It was not. The instability was the shadowing bug corrupting
the lateral trajectory. Once that was fixed, per-axis and norm estimates
agreed to within 0.16. The test now asserts agreement, and the earlier
belief is recorded in the docstring so the next person does not re-derive
the wrong conclusion.

## Limits

- Flat-Earth NED. No curvature, no rotating frame, no Coriolis. Fine over
  the ranges here, wrong for anything orbital.
- Aerodynamic coefficients are synthetic. They are shaped to be physically
  sensible (transonic drag rise, negative static margin, damping in all
  three axes) but they do not represent any real vehicle, and nothing here
  has been validated against flight data.
- No propulsion model beyond a constant thrust term. No mass depletion.
- Linear interpolation on the coefficient tables, clamped rather than
  extrapolated at the edges. Clamping is deliberate: a linear extrapolation
  off the end of the Mach table gives negative drag and a vehicle that
  gains energy from nowhere.
- Single active controller. There is no guidance layer, no navigation
  filter, and the "flight software" is a PID.

## Running it

```
python3 tests/run.py     # 85 checks
python3 bench.py         # writes bench-results.json
```

