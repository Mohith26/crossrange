"""Vehicle description: mass properties, aerodynamics, and actuators.

The aero model is a coefficient build-up interpolated on Mach and angle of
attack. Real programmes get these tables from CFD and wind tunnel runs and
then correct them against flight data; the numbers here are synthetic and
shaped to be physically sensible rather than to represent any real vehicle.
That distinction matters enough that it is repeated in the README.

Two decisions worth calling out:

Interpolation is linear and clamped at the table edges. Beyond the edge the
coefficient is held flat rather than extrapolated, because a linear
extrapolation off the end of a Mach table produces negative drag around
Mach 6 and a trajectory that gains energy out of nowhere.

Actuators are modelled as first-order lags with rate and position limits.
Leaving them out entirely is the single most common way a simulator makes a
control law look better than it is, since an ideal surface can move faster
than any real servo.
"""

import math


def _interp1(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo = 0
    hi = len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    t = 0.0 if span == 0 else (x - xs[lo]) / span
    return ys[lo] + t * (ys[hi] - ys[lo])


class Actuator:
    """First-order lag with rate and deflection limits.

    state' = clamp_rate((command - state) / tau)
    """

    def __init__(self, tau=0.05, rate_limit=math.radians(200.0), limit=math.radians(25.0)):
        self.tau = tau
        self.rate_limit = rate_limit
        self.limit = limit

    def derivative(self, state, command):
        cmd = max(-self.limit, min(self.limit, command))
        rate = (cmd - state) / self.tau
        return max(-self.rate_limit, min(self.rate_limit, rate))


class Vehicle:
    def __init__(self, **overrides):
        # Mass properties. Ixy and Iyz are assumed zero (symmetric vehicle);
        # Ixz is kept because it is the cross term that actually matters for
        # a slender airframe and dropping it hides roll-yaw coupling.
        self.mass = 1200.0            # kg
        self.ixx = 1200.0             # kg m^2
        self.iyy = 8500.0
        self.izz = 9000.0
        self.ixz = 150.0

        self.s_ref = 1.8              # m^2, reference area
        self.c_bar = 1.6              # m, reference chord (pitch)
        self.b_ref = 3.2              # m, reference span (roll/yaw)

        # Static coefficients versus Mach.
        self.mach_grid = [0.0, 0.6, 0.9, 1.05, 1.2, 2.0, 3.0, 4.0, 5.0]
        self.cd0_grid = [0.021, 0.023, 0.038, 0.079, 0.071, 0.052, 0.043, 0.039, 0.037]
        self.cla_grid = [3.10, 3.35, 4.05, 3.60, 3.15, 2.35, 1.90, 1.66, 1.52]
        self.cma_grid = [-0.62, -0.66, -0.79, -0.90, -0.84, -0.63, -0.51, -0.45, -0.41]

        self.k_induced = 0.13         # drag polar: CD = CD0 + k CL^2
        self.cm_q = -12.5             # pitch damping, per rad
        self.cl_p = -0.42             # roll damping
        self.cn_r = -0.28             # yaw damping
        self.cy_beta = -0.85
        self.cn_beta = 0.14           # weathercock stability
        self.cl_beta = -0.09          # dihedral effect

        self.cm_de = -0.92            # elevator power
        self.cl_da = 0.14             # aileron power
        self.cn_dr = -0.07            # rudder power

        self.thrust_max = 0.0         # N, unpowered glide by default

        self.elevator = Actuator()
        self.aileron = Actuator(tau=0.04)
        self.rudder = Actuator(tau=0.06)

        for k, v in overrides.items():
            if not hasattr(self, k):
                raise AttributeError("Vehicle has no parameter %r" % k)
            setattr(self, k, v)

    def inertia(self):
        return self.ixx, self.iyy, self.izz, self.ixz

    def coefficients(self, air, rates, controls):
        """Body-axis force and moment coefficients.

        air: dict from frames.airdata
        rates: (p, q, r) body rates, rad/s
        controls: (elevator, aileron, rudder) deflections, rad
        """
        mach = air["mach"]
        alpha = air["alpha"]
        beta = air["beta"]
        vt = air["vt"]
        p, q, r = rates
        de, da, dr = controls

        cd0 = _interp1(self.mach_grid, self.cd0_grid, mach)
        cla = _interp1(self.mach_grid, self.cla_grid, mach)
        cma = _interp1(self.mach_grid, self.cma_grid, mach)

        cl = cla * alpha
        cd = cd0 + self.k_induced * cl * cl
        cy = self.cy_beta * beta

        # Non-dimensional rate terms. Guarded against the zero-airspeed case
        # so that a vehicle sitting still does not divide by zero.
        if vt > 1.0:
            qhat = q * self.c_bar / (2.0 * vt)
            phat = p * self.b_ref / (2.0 * vt)
            rhat = r * self.b_ref / (2.0 * vt)
        else:
            qhat = phat = rhat = 0.0

        cm = cma * alpha + self.cm_q * qhat + self.cm_de * de
        croll = self.cl_beta * beta + self.cl_p * phat + self.cl_da * da
        cn = self.cn_beta * beta + self.cn_r * rhat + self.cn_dr * dr

        # Lift and drag live in the wind frame; rotate them into body axes.
        ca = math.cos(alpha)
        sa = math.sin(alpha)
        cx = -cd * ca + cl * sa
        cz = -cd * sa - cl * ca
        return dict(cx=cx, cy=cy, cz=cz, cl=croll, cm=cm, cn=cn, lift=cl, drag=cd)

