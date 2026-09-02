"""Reference frames, attitude, and the standard atmosphere.

Attitude is carried as a quaternion rather than Euler angles. Euler angles
are easier to read in a log file and they are what the HMI wants to show a
pilot, but they gimbal lock at 90 degrees of pitch, and a vehicle that
pitches through vertical during a boost is exactly the case this simulator
has to survive. Euler angles are derived on the way out instead.

Quaternion convention here is scalar-first, [w, x, y, z], rotating a vector
from body axes into the local NED frame. Body axes are the usual aircraft
set: x forward out the nose, y out the right wing, z down.
"""

import math

# WGS-84-ish constants. Flat-Earth NED is fine for the ranges this sim covers;
# the curvature error over a few hundred kilometres is smaller than the
# uncertainty in the aero tables it is fed.
G0 = 9.80665            # m/s^2, standard gravity
R_AIR = 287.05287       # J/(kg K), specific gas constant for dry air
GAMMA_AIR = 1.4         # ratio of specific heats


def quat_normalize(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("cannot normalize a zero quaternion")
    return (w / n, x / n, y / n, z / n)


def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_from_euler(roll, pitch, yaw):
    """ZYX (yaw, then pitch, then roll) body-from-NED, angles in radians."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return quat_normalize((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ))


def euler_from_quat(q):
    """Returns (roll, pitch, yaw) in radians.

    Pitch uses a clamped asin because a quaternion that has drifted by one
    part in 1e-15 can push the argument just past 1.0 and raise, which is a
    miserable way to lose a Monte Carlo case.
    """
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def body_to_ned(q, v):
    """Rotate a body-frame vector into NED using the quaternion sandwich."""
    w, x, y, z = q
    vx, vy, vz = v
    # Expanded q * v * q_conj. Written out rather than composed from
    # quat_multiply because this runs inside the derivative evaluation and
    # gets called four times per integration step.
    t2 = w * x
    t3 = w * y
    t4 = w * z
    t5 = -x * x
    t6 = x * y
    t7 = x * z
    t8 = -y * y
    t9 = y * z
    t10 = -z * z
    return (
        2.0 * ((t8 + t10) * vx + (t6 - t4) * vy + (t3 + t7) * vz) + vx,
        2.0 * ((t4 + t6) * vx + (t5 + t10) * vy + (t9 - t2) * vz) + vy,
        2.0 * ((t7 - t3) * vx + (t2 + t9) * vy + (t5 + t8) * vz) + vz,
    )


def ned_to_body(q, v):
    w, x, y, z = q
    return body_to_ned((w, -x, -y, -z), v)


# --- 1976 standard atmosphere, troposphere through the lower stratosphere ---
# Layer table: (base altitude m, base temperature K, lapse rate K/m, base pressure Pa)
_LAYERS = [
    (0.0, 288.15, -0.0065, 101325.0),
    (11000.0, 216.65, 0.0, 22632.06),
    (20000.0, 216.65, 0.001, 5474.889),
    (32000.0, 228.65, 0.0028, 868.0187),
    (47000.0, 270.65, 0.0, 110.9063),
]


def atmosphere(altitude_m):
    """Returns (temperature K, pressure Pa, density kg/m^3, speed of sound m/s).

    Above the top of the table the model is held at the 51 km values rather
    than extrapolated. That is wrong physics, but it is wrong in a loud and
    obvious way, which beats silently returning a negative density.
    """
    h = max(0.0, min(altitude_m, 51000.0))
    base_h, base_t, lapse, base_p = _LAYERS[0]
    for layer in _LAYERS:
        if h >= layer[0]:
            base_h, base_t, lapse, base_p = layer
        else:
            break

    dh = h - base_h
    if lapse == 0.0:
        temp = base_t
        press = base_p * math.exp(-G0 * dh / (R_AIR * base_t))
    else:
        temp = base_t + lapse * dh
        press = base_p * (temp / base_t) ** (-G0 / (lapse * R_AIR))

    rho = press / (R_AIR * temp)
    a = math.sqrt(GAMMA_AIR * R_AIR * temp)
    return temp, press, rho, a


def airdata(velocity_body, altitude_m, wind_ned=(0.0, 0.0, 0.0), q=None):
    """Airspeed, Mach, dynamic pressure, angle of attack, sideslip.

    Wind arrives in NED because that is how it is measured and how a gust
    model is specified; it has to be rotated into body axes before it can be
    subtracted from the body-frame velocity.
    """
    temp, press, rho, a = atmosphere(altitude_m)
    u, v, w = velocity_body
    if q is not None and any(wind_ned):
        wu, wv, ww = ned_to_body(q, wind_ned)
        u, v, w = u - wu, v - wv, w - ww

    vt = math.sqrt(u * u + v * v + w * w)
    mach = vt / a if a > 0 else 0.0
    qbar = 0.5 * rho * vt * vt

    # Alpha and beta are undefined at zero airspeed; returning zeros keeps
    # the derivative finite on the launch rail instead of producing a NaN
    # that silently poisons the whole trajectory.
    if vt < 1e-6:
        return dict(vt=0.0, mach=0.0, qbar=0.0, alpha=0.0, beta=0.0, rho=rho, a=a)

    alpha = math.atan2(w, u)
    beta = math.asin(max(-1.0, min(1.0, v / vt)))
    return dict(vt=vt, mach=mach, qbar=qbar, alpha=alpha, beta=beta, rho=rho, a=a)

