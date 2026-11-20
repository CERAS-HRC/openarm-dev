#!/usr/bin/env python3
"""Slowly sweep one RobStride motor +90 deg and back a few times, using MIT commands.

Bring the interface up first (RobStride default is 1 Mbps):

    sudo ip link set can0 up type can bitrate 1000000

Run:

    python3 test_sweep.py --id 1 --model RS03
    python3 test_sweep.py --id 1 --model RS03 --amplitude -90 --cycles 5 --period 8

The sweep is relative to wherever the motor is when the script starts, and it uses
a smooth cosine profile with matching velocity feed-forward. Ctrl+C disables the
motor (it goes limp), so make sure the load is safe to drop.
"""

from __future__ import annotations

import argparse
import math
import time

from robstride import MODEL_LIMITS, RobStrideBus, RobStrideError


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--id", type=int, default=127, help="motor CAN id (factory default is often 127)")
    ap.add_argument("--host-id", type=lambda s: int(s, 0), default=0xFD)
    ap.add_argument("--model", default="RS03", choices=sorted(MODEL_LIMITS))
    ap.add_argument("--amplitude", type=float, default=90.0, help="sweep size in degrees (sign = direction)")
    ap.add_argument("--cycles", type=int, default=3, help="number of out-and-back cycles")
    ap.add_argument("--period", type=float, default=6.0, help="seconds per out-and-back cycle")
    ap.add_argument("--kp", type=float, default=15.0)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--rate", type=float, default=200.0, help="command rate in Hz")
    ap.add_argument("--feedback-timeout", type=float, default=0.5,
                    help="abort if no motor feedback for this many seconds")
    args = ap.parse_args()

    amp = math.radians(args.amplitude)
    dt = 1.0 / args.rate

    with RobStrideBus(args.channel, host_id=args.host_id) as bus:
        bus.add_motor(args.id, args.model)

        fb = bus.ping(args.id)
        if fb is None:
            print(f"No reply from motor {args.id} on {args.channel}. Check wiring, power, id and bitrate.")
            return 1
        print(f"Found motor {args.id}: pos={math.degrees(fb.position):.1f} deg "
              f"temp={fb.temperature:.1f} C mode={fb.mode_name} faults={fb.faults or 'none'}")

        try:
            fb = bus.enable(args.id)
        except RobStrideError as e:
            print(f"Enable failed: {e}")
            return 1

        p0 = fb.position
        p_max = bus.limits(args.id).p_max
        if not (-p_max < p0 + amp < p_max and -p_max < p0 < p_max):
            print(f"Sweep {math.degrees(p0):.1f} -> {math.degrees(p0 + amp):.1f} deg exceeds the "
                  f"+-{math.degrees(p_max):.0f} deg MIT range. Re-zero the motor first.")
            bus.disable(args.id)
            return 1

        print(f"Enabled. Sweeping {args.amplitude:+.0f} deg from {math.degrees(p0):.1f} deg, "
              f"{args.cycles} cycles, {args.period:.1f} s each (kp={args.kp}, kd={args.kd}).")

        total = args.cycles * args.period
        w = 2 * math.pi / args.period
        last_print = 0.0
        try:
            t_start = time.monotonic()
            next_tick = t_start
            while True:
                t = time.monotonic() - t_start
                if t >= total:
                    break
                # Cosine profile: 0 -> amp -> 0 each period, zero velocity at the ends.
                p_des = p0 + amp * 0.5 * (1 - math.cos(w * t))
                v_des = amp * 0.5 * w * math.sin(w * t)
                # Fire-and-forget so USB/bridge latency doesn't stall the loop;
                # replies are picked up by the RX thread and read from the cache.
                bus.mit(args.id, p=p_des, v=v_des, kp=args.kp, kd=args.kd, t=0.0, wait=False)
                fb = bus.feedback(args.id)
                age = time.monotonic() - fb.stamp if fb else float("inf")
                if age > args.feedback_timeout:
                    raise RobStrideError(f"no feedback from motor for {age:.2f} s")
                if fb is not None:
                    if fb.faults:
                        raise RobStrideError(f"motor fault: {fb.faults}")
                    if t - last_print >= 0.25:
                        last_print = t
                        print(f"t={t:5.2f}s  cmd={math.degrees(p_des - p0):7.2f}  "
                              f"pos={math.degrees(fb.position - p0):7.2f} deg  "
                              f"vel={fb.velocity:6.2f} rad/s  tau={fb.torque:6.2f} Nm  "
                              f"T={fb.temperature:.0f}C")
                next_tick += dt
                sleep = next_tick - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.monotonic()

            # Hold the start position briefly before letting go.
            hold_end = time.monotonic() + 0.5
            while time.monotonic() < hold_end:
                bus.mit(args.id, p=p0, v=0.0, kp=args.kp, kd=args.kd, wait=False)
                time.sleep(dt)
            print("Sweep complete.")
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except RobStrideError as e:
            print(f"Error: {e}")
        finally:
            bus.disable(args.id)
            print("Motor disabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
