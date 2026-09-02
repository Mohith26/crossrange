"""The equations of motion.

State vector, 16 elements, in this order:

    0:3   position NED (m)
    3:6   velocity body (m/s)      u, v, w
    6:10  attitude quaternion      w, x, y, z
    10:13 body rates (rad/s)       p, q, r
    13:16 actuator states (rad)    elevator, aileron, rudder

Velocity is carried in body axes because that is where the aerodynamic
forces are computed, which avoids rotating the force vector every step.
The cost is the transport term omega x v in the translational equation,
which is easy to forget and produces a subtly wrong turn rate when you do.
"""

import math

from .frames import G0, body_to_ned, ned_to_body, airdata, quat_normalize


def derivative(t, state, vehicle, commands, wind_ned=(0.0, 0.0, 0.0), thrust=0.0):
    pn, pe, pd = state[0], state[1], state[2]
    u, v, w = state[3], state[4], state[5]
    q0, q1, q2, q3 = state[6], state[7], state[8], state[9]
    p, qq, r = state[10], state[11], state[12]
    de, da, dr = state[13], state[14], state[15]

    quat = (q0, q1, q2, q3)
    altitude = -pd  # NED: down is positive, so altitude is negative pd

    air = airdata((u, v, w), altitude, wind_ned, quat)
    coef = vehicle.coefficients(air, (p, qq, r), (de, da, dr))

    qbar_s = air["qbar"] * vehicle.s_ref
    fx = qbar_s * coef["cx"] + thrust
    fy = qbar_s * coef["cy"]
    fz = qbar_s * coef["cz"]

    # Gravity is defined in NED and has to come into body axes.
    gx, gy, gz = ned_to_body(quat, (0.0, 0.0, G0))

    # Translational: v_dot = F/m + g_body - omega x v
    du = fx / vehicle.mass + gx - (qq * w - r * v)
    dv = fy / vehicle.mass + gy - (r * u - p * w)
    dw = fz / vehicle.mass + gz - (p * v - qq * u)

    # Position derivative is body velocity rotated into NED.
    dpn, dpe, dpd = body_to_ned(quat, (u, v, w))

    # Attitude: q_dot = 0.5 * q (x) omega_quaternion
    dq0 = 0.5 * (-q1 * p - q2 * qq - q3 * r)
    dq1 = 0.5 * (q0 * p + q2 * r - q3 * qq)
    dq2 = 0.5 * (q0 * qq - q1 * r + q3 * p)
    dq3 = 0.5 * (q0 * r + q1 * qq - q2 * p)

    # Rotational: I omega_dot = M - omega x (I omega), solved for a body with
    # an Ixz product of inertia. The roll and yaw equations are coupled and
    # get solved together; treating them as independent is a classic error
    # that shows up as the wrong spiral mode.
    ixx, iyy, izz, ixz = vehicle.inertia()
    l_mom = qbar_s * vehicle.b_ref * coef["cl"]
    m_mom = qbar_s * vehicle.c_bar * coef["cm"]
    n_mom = qbar_s * vehicle.b_ref * coef["cn"]

    # These are named with an explicit _dt suffix because the obvious short
    # names collide with the control deflections unpacked above. Calling the
    # yaw acceleration `dr` shadowed the rudder deflection `dr`, so the
    # rudder actuator was handed an angular acceleration as its current
    # position. A 5 degree aileron step then produced 204 deg/s of yaw rate
    # and rolled the vehicle the wrong way, and every case with any lateral
    # input diverged. The pitch-only no-wind case looked fine purely because
    # r stayed near zero there, which is what made it hard to spot.
    gamma = ixx * izz - ixz * ixz
    pq_term = l_mom + ixz * p * qq - (izz - iyy) * qq * r
    qr_term = n_mom - ixz * qq * r - (iyy - ixx) * p * qq
    dp_dt = (izz * pq_term + ixz * qr_term) / gamma
    dr_dt = (ixz * pq_term + ixx * qr_term) / gamma
    dq_dt = (m_mom - (ixx - izz) * p * r - ixz * (p * p - r * r)) / iyy

    # Actuators chase their commands. de/da/dr here are the deflections from
    # the state vector, untouched by the block above.
    dde = vehicle.elevator.derivative(de, commands[0])
    dda = vehicle.aileron.derivative(da, commands[1])
    ddr = vehicle.rudder.derivative(dr, commands[2])

    return [dpn, dpe, dpd, du, dv, dw, dq0, dq1, dq2, dq3,
            dp_dt, dq_dt, dr_dt, dde, dda, ddr]


def make_state(position=(0.0, 0.0, -1000.0), velocity=(120.0, 0.0, 0.0),
               quat=(1.0, 0.0, 0.0, 0.0), rates=(0.0, 0.0, 0.0),
               actuators=(0.0, 0.0, 0.0)):
    return list(position) + list(velocity) + list(quat_normalize(quat)) + list(rates) + list(actuators)


def renormalize(state):
    """Pull the quaternion back onto the unit sphere.

    RK4 does not preserve the norm, and the drift compounds. Renormalizing
    every step is cheap and keeps the attitude honest; the test suite
    measures how far it drifts before this is applied.
    """
    q = quat_normalize((state[6], state[7], state[8], state[9]))
    state[6], state[7], state[8], state[9] = q
    return state


def energy(state, vehicle):
    """Specific mechanical energy, used as a conservation check."""
    u, v, w = state[3], state[4], state[5]
    speed_sq = u * u + v * v + w * w
    altitude = -state[2]
    return 0.5 * speed_sq + G0 * altitude

