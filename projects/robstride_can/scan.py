#!/usr/bin/env python3
"""Find RobStride motors on the bus by pinging each CAN id. Motors do not move.

    python3 scan.py --channel vcan0
    python3 scan.py --channel vcan0 --ids 1-20
"""

from __future__ import annotations

import argparse
import math

from robstride import MODEL_LIMITS, RobStrideBus


def parse_ids(spec: str) -> list[int]:
    ids: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        ids.extend(range(int(lo), int(hi or lo) + 1))
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--host-id", type=lambda s: int(s, 0), default=0xFD)
    ap.add_argument("--ids", default="1-127", help="e.g. 1-127 or 1,2,5-8")
    ap.add_argument("--timeout", type=float, default=0.1, help="seconds to wait for each reply")
    ap.add_argument("--model", default="RS03", choices=sorted(MODEL_LIMITS),
                    help="only affects how position/temperature are decoded")
    args = ap.parse_args()

    found = []
    with RobStrideBus(args.channel, host_id=args.host_id) as bus:
        for mid in parse_ids(args.ids):
            bus.add_motor(mid, args.model)
            fb = bus.ping(mid, timeout=args.timeout)
            if fb is not None:
                found.append(mid)
                print(f"  id {mid:3d}: pos={math.degrees(fb.position):7.1f} deg  temp={fb.temperature:.1f} C  "
                      f"mode={fb.mode_name}  faults={fb.faults or 'none'}")
    if not found:
        print("No motors answered. Check: bridge running, motor powered, CAN H/L wired, "
              "120 ohm termination, bitrate, and try --host-id 0.")
        return 1
    print(f"Found {len(found)} motor(s): {found}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
