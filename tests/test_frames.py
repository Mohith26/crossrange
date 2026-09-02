"""Frames and atmosphere, checked against published values.

The ISA numbers here are the tabulated 1976 standard atmosphere values, not
numbers this code produced. If the implementation drifts, these fail.
"""

import math

from sixdof.frames import (atmosphere, airdata, quat_from_euler, euler_from_quat,
                           quat_multiply, quat_normalize, body_to_ned, ned_to_body)


def register(check, close):
    def test_sea_level():
        t, p, rho, a = atmosphere(0.0)
        close("ISA sea level temperature", t, 288.15, 0.01)
        close("ISA sea level pressure", p, 101325.0, 1.0)
        close("ISA sea level density", rho, 1.2250, 0.001)
        close("ISA sea level speed of sound", a, 340.29, 0.05)

    def test_tropopause():
        t, p, rho, a = atmosphere(11000.0)
        close("ISA 11 km temperature", t, 216.65, 0.01)
        close("ISA 11 km pressure", p, 22632.1, 5.0)
        close("ISA 11 km density", rho, 0.36392, 0.0005)
        close("ISA 11 km speed of sound", a, 295.07, 0.05)

    def test_stratosphere():
        t, p, rho, a = atmosphere(20000.0)
        close("ISA 20 km temperature", t, 216.65, 0.01)
        close("ISA 20 km pressure", p, 5474.9, 2.0)
        t2, p2, _, _ = atmosphere(32000.0)
        close("ISA 32 km temperature", t2, 228.65, 0.05)
        close("ISA 32 km pressure", p2, 868.02, 1.0)

    def test_atmosphere_monotonic():
        prev_p = 1e9
        prev_rho = 1e9
        ok = True
        for h in range(0, 50001, 500):
            _, p, rho, _ = atmosphere(float(h))
            if p >= prev_p or rho >= prev_rho:
                ok = False
                break
            prev_p, prev_rho = p, rho
        check("pressure and density fall monotonically to 50 km", ok)

    def test_atmosphere_clamped():
        a = atmosphere(-500.0)
        b = atmosphere(0.0)
        check("below sea level clamps rather than extrapolating", a == b)
        c = atmosphere(80000.0)
        d = atmosphere(51000.0)
        check("above the table clamps rather than going negative", c == d and c[2] > 0)

    def test_euler_roundtrip():
        worst = 0.0
        for roll_d in (-170, -90, -45, 0, 30, 89, 175):
            for pitch_d in (-85, -45, 0, 20, 60, 85):
                for yaw_d in (-179, -90, 0, 45, 178):
                    e = (math.radians(roll_d), math.radians(pitch_d), math.radians(yaw_d))
                    back = euler_from_quat(quat_from_euler(*e))
                    for i in range(3):
                        diff = abs(back[i] - e[i])
                        diff = min(diff, abs(2 * math.pi - diff))
                        worst = max(worst, diff)
        check("euler to quaternion round trip within 1e-9 rad over 210 attitudes (worst %.2e)" % worst,
              worst < 1e-9)

    def test_quaternion_norm():
        q = quat_from_euler(0.3, -0.7, 1.9)
        n = math.sqrt(sum(c * c for c in q))
        close("quaternion is unit norm", n, 1.0, 1e-12)

    def test_rotation_known_vector():
        # 90 degrees of yaw should send body x onto NED +y (east).
        q = quat_from_euler(0.0, 0.0, math.radians(90.0))
        v = body_to_ned(q, (1.0, 0.0, 0.0))
        close("yaw 90: north component", v[0], 0.0, 1e-12)
        close("yaw 90: east component", v[1], 1.0, 1e-12)
        # 90 degrees nose-up should send body x onto NED -z (straight up).
        q2 = quat_from_euler(0.0, math.radians(90.0), 0.0)
        v2 = body_to_ned(q2, (1.0, 0.0, 0.0))
        close("pitch 90: down component is -1", v2[2], -1.0, 1e-12)

    def test_rotation_inverse():
        q = quat_from_euler(0.4, 0.9, -2.1)
        v = (12.0, -3.5, 7.25)
        back = ned_to_body(q, body_to_ned(q, v))
        worst = max(abs(back[i] - v[i]) for i in range(3))
        check("body to NED and back is the identity (worst %.2e)" % worst, worst < 1e-12)

    def test_rotation_preserves_length():
        q = quat_from_euler(-1.2, 0.55, 2.7)
        v = (3.0, -4.0, 12.0)
        n0 = math.sqrt(sum(c * c for c in v))
        r = body_to_ned(q, v)
        n1 = math.sqrt(sum(c * c for c in r))
        close("rotation preserves vector length", n1, n0, 1e-12)

    def test_quat_multiply_associates():
        a = quat_from_euler(0.2, 0.3, 0.4)
        b = quat_from_euler(-0.5, 0.1, 1.1)
        c = quat_from_euler(0.9, -0.2, -0.7)
        left = quat_multiply(quat_multiply(a, b), c)
        right = quat_multiply(a, quat_multiply(b, c))
        worst = max(abs(left[i] - right[i]) for i in range(4))
        check("quaternion multiplication is associative (worst %.2e)" % worst, worst < 1e-12)

    def test_airdata_alpha_beta():
        # Pure forward flight: alpha and beta are both zero.
        ad = airdata((200.0, 0.0, 0.0), 5000.0)
        close("level flight alpha", ad["alpha"], 0.0, 1e-12)
        close("level flight beta", ad["beta"], 0.0, 1e-12)
        # w positive (flow up through the belly) is positive alpha.
        ad2 = airdata((200.0, 0.0, 20.0), 5000.0)
        check("upward w gives positive alpha", ad2["alpha"] > 0)
        close("alpha equals atan(w/u)", ad2["alpha"], math.atan2(20.0, 200.0), 1e-12)
        ad3 = airdata((200.0, 15.0, 0.0), 5000.0)
        check("positive v gives positive beta", ad3["beta"] > 0)

    def test_airdata_mach_and_qbar():
        h = 10000.0
        _, _, rho, a = atmosphere(h)
        v = 250.0
        ad = airdata((v, 0.0, 0.0), h)
        close("mach is airspeed over speed of sound", ad["mach"], v / a, 1e-12)
        close("dynamic pressure is half rho v squared", ad["qbar"], 0.5 * rho * v * v, 1e-9)

    def test_airdata_zero_speed_is_finite():
        ad = airdata((0.0, 0.0, 0.0), 1000.0)
        check("zero airspeed returns finite alpha/beta rather than NaN",
              ad["alpha"] == 0.0 and ad["beta"] == 0.0 and ad["qbar"] == 0.0)

    def test_wind_subtracts():
        q = quat_from_euler(0.0, 0.0, 0.0)
        still = airdata((200.0, 0.0, 0.0), 3000.0, (0.0, 0.0, 0.0), q)
        tail = airdata((200.0, 0.0, 0.0), 3000.0, (30.0, 0.0, 0.0), q)
        close("a 30 m/s tailwind removes 30 m/s of airspeed", tail["vt"], still["vt"] - 30.0, 1e-9)
        head = airdata((200.0, 0.0, 0.0), 3000.0, (-30.0, 0.0, 0.0), q)
        close("a 30 m/s headwind adds 30 m/s of airspeed", head["vt"], still["vt"] + 30.0, 1e-9)

    return [test_sea_level, test_tropopause, test_stratosphere, test_atmosphere_monotonic,
            test_atmosphere_clamped, test_euler_roundtrip, test_quaternion_norm,
            test_rotation_known_vector, test_rotation_inverse, test_rotation_preserves_length,
            test_quat_multiply_associates, test_airdata_alpha_beta, test_airdata_mach_and_qbar,
            test_airdata_zero_speed_is_finite, test_wind_subtracts]

