#!/usr/bin/env python3
"""Bridge a Waveshare USB-CAN-A adapter to a Linux SocketCAN interface (vcan).

The USB-CAN-A is a CH340 USB-serial chip in front of a CAN controller and speaks
Waveshare's own serial framing, so Linux never creates a ``can0`` for it. This
script copies frames both ways between the serial port and a virtual CAN
interface, so SocketCAN code (robstride.py, candump, ...) can use ``vcan0``.

One-time setup per boot (needs sudo)::

    sudo modprobe vcan
    sudo ip link add vcan0 type vcan
    sudo ip link set vcan0 up

Then leave this running::

    python3 waveshare_bridge.py                     # /dev/ttyUSB0 <-> vcan0 @ 1 Mbps
    python3 waveshare_bridge.py --selftest          # adapter loopback check, no CAN bus needed

Waveshare serial protocol (serial at 2 Mbaud, 8 data bits, no parity):

  Settings frame, 20 bytes:
    AA 55 12 <speed> <frame type> <filter id x4> <mask id x4> <mode> 01 00 00 00 00 <checksum>
    checksum = sum(bytes[2:19]) & 0xFF

  Variable-length data frame:
    AA <info> <id, little-endian: 2 bytes std / 4 bytes ext> <data x DLC> 55
    info = 0xC0 | (0x20 if extended) | (0x10 if remote) | DLC
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import threading
import time

try:
    import serial
except ImportError:  # pragma: no cover
    sys.exit("pyserial is required: sudo apt install python3-serial  (or pip install pyserial)")

SPEED_CODES = {
    1_000_000: 0x01, 800_000: 0x02, 500_000: 0x03, 400_000: 0x04, 250_000: 0x05, 200_000: 0x06,
    125_000: 0x07, 100_000: 0x08, 50_000: 0x09, 20_000: 0x0A, 10_000: 0x0B, 5_000: 0x0C,
}
MODES = {"normal": 0x00, "loopback": 0x01, "silent": 0x02, "loopback-silent": 0x03}
FRAME_STD, FRAME_EXT = 0x01, 0x02

_CAN_FRAME = struct.Struct("=IB3x8s")
CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_EFF_MASK = 0x1FFFFFFF
CAN_SFF_MASK = 0x000007FF


# ---------------------------------------------------------------------------
# Waveshare protocol
# ---------------------------------------------------------------------------


def settings_frame(bitrate: int, mode: str = "normal", extended: bool = True) -> bytes:
    if bitrate not in SPEED_CODES:
        raise ValueError(f"unsupported bitrate {bitrate}; choose from {sorted(SPEED_CODES)}")
    body = bytes([0x12, SPEED_CODES[bitrate], FRAME_EXT if extended else FRAME_STD])
    body += bytes(4) + bytes(4)  # filter id / mask = 0 -> accept everything
    body += bytes([MODES[mode], 0x01, 0, 0, 0, 0])
    return b"\xAA\x55" + body + bytes([sum(body) & 0xFF])


def encode_frame(can_id: int, data: bytes, extended: bool, remote: bool = False) -> bytes:
    dlc = len(data)
    if dlc > 8:
        raise ValueError("classic CAN frames carry at most 8 bytes")
    info = 0xC0 | dlc | (0x20 if extended else 0) | (0x10 if remote else 0)
    ident = struct.pack("<I", can_id & CAN_EFF_MASK) if extended else struct.pack("<H", can_id & CAN_SFF_MASK)
    return bytes([0xAA, info]) + ident + data + b"\x55"


class FrameParser:
    """Incremental parser for the adapter's serial byte stream."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.dropped = 0

    def feed(self, chunk: bytes) -> list[tuple[int, bytes, bool, bool]]:
        """Return a list of (can_id, data, extended, remote)."""
        self.buf += chunk
        out = []
        while True:
            start = self.buf.find(0xAA)
            if start < 0:
                self.dropped += len(self.buf)
                self.buf.clear()
                break
            if start:
                self.dropped += start
                del self.buf[:start]
            if len(self.buf) < 2:
                break
            info = self.buf[1]
            if info == 0x55:  # 20-byte fixed/settings frame: skip it
                if len(self.buf) < 20:
                    break
                del self.buf[:20]
                continue
            if info & 0xC0 != 0xC0:  # not a frame start, resync
                self.dropped += 1
                del self.buf[:1]
                continue
            extended = bool(info & 0x20)
            remote = bool(info & 0x10)
            dlc = info & 0x0F
            id_len = 4 if extended else 2
            total = 2 + id_len + dlc + 1
            if dlc > 8:
                self.dropped += 1
                del self.buf[:1]
                continue
            if len(self.buf) < total:
                break
            if self.buf[total - 1] != 0x55:  # bad terminator, resync
                self.dropped += 1
                del self.buf[:1]
                continue
            ident = int.from_bytes(self.buf[2 : 2 + id_len], "little")
            data = bytes(self.buf[2 + id_len : 2 + id_len + dlc])
            del self.buf[:total]
            out.append((ident, data, extended, remote))
        return out


def open_adapter(port: str, bitrate: int, mode: str, serial_baud: int, stopbits: int) -> serial.Serial:
    try:
        ser = serial.Serial(
            port, serial_baud, bytesize=8, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_TWO if stopbits == 2 else serial.STOPBITS_ONE, timeout=0.01,
        )
    except serial.SerialException as e:
        if "Permission denied" in str(e):
            sys.exit(f"Permission denied on {port}. Run: sudo usermod -aG dialout $USER  "
                     f"then log out/in, or open a fresh login shell with: su - $USER")
        raise
    ser.reset_input_buffer()
    ser.write(settings_frame(bitrate, mode))
    ser.flush()
    time.sleep(0.1)  # let the adapter apply settings
    ser.reset_input_buffer()
    return ser


# ---------------------------------------------------------------------------
# Self-test: adapter internal loopback, no bus or motor needed
# ---------------------------------------------------------------------------


def selftest(args) -> int:
    ser = open_adapter(args.port, args.bitrate, "loopback", args.serial_baud, args.stopbits)
    parser = FrameParser()
    # Extended frames are what RobStride uses, so only they are required to pass.
    # The adapter is configured for extended frames and may filter standard ones.
    tests = [(0x1234567, bytes(range(8)), True, True), (0x123, b"\x11\x22\x33", False, False)]
    ok = True
    for can_id, data, ext, required in tests:
        got = []
        for _attempt in range(3):
            ser.write(encode_frame(can_id, data, ext))
            deadline = time.monotonic() + 0.3
            while time.monotonic() < deadline and not got:
                got = parser.feed(ser.read(64))
            if got:
                break
        kind = "ext" if ext else "std"
        if got and got[0][0] == can_id and got[0][1] == data:
            print(f"  {kind} frame 0x{can_id:X}: OK")
        elif required:
            ok = False
            print(f"  {kind} frame 0x{can_id:X}: FAILED (got {got!r}, dropped {parser.dropped} bytes)")
        else:
            print(f"  {kind} frame 0x{can_id:X}: no echo (fine: RobStride uses extended frames only)")
    # Return the adapter to normal mode so it's ready for the bridge.
    ser.write(settings_frame(args.bitrate, "normal"))
    ser.flush()
    ser.close()
    print("Self-test passed: adapter is talking." if ok else
          "Self-test failed. Try --stopbits 1, replug the adapter, or check the port.")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


def bridge(args) -> int:
    try:
        sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        sock.bind((args.iface,))
    except OSError as e:
        sys.exit(f"Can't open {args.iface} ({e}). Create it first:\n"
                 f"  sudo modprobe vcan && sudo ip link add {args.iface} type vcan && sudo ip link set {args.iface} up")
    ser = open_adapter(args.port, args.bitrate, args.mode, args.serial_baud, args.stopbits)
    print(f"Bridging {args.port} <-> {args.iface} at {args.bitrate} bit/s ({args.mode}). Ctrl+C to stop.")

    stats = {"to_bus": 0, "from_bus": 0}
    stop = threading.Event()

    def serial_to_can() -> None:
        parser = FrameParser()
        while not stop.is_set():
            try:
                chunk = ser.read(ser.in_waiting or 1)
            except serial.SerialException as e:
                print(f"Serial error: {e}")
                stop.set()
                break
            for can_id, data, ext, rtr in parser.feed(chunk):
                cid = can_id | (CAN_EFF_FLAG if ext else 0) | (CAN_RTR_FLAG if rtr else 0)
                sock.send(_CAN_FRAME.pack(cid, len(data), data.ljust(8, b"\x00")))
                stats["from_bus"] += 1
                if args.verbose:
                    print(f"  bus -> {args.iface}: {can_id:08X} [{len(data)}] {data.hex(' ')}")

    def can_to_serial() -> None:
        sock.settimeout(0.2)
        while not stop.is_set():
            try:
                raw = sock.recv(16)
            except socket.timeout:
                continue
            except OSError:
                break
            cid, dlc, data = _CAN_FRAME.unpack(raw)
            if cid & CAN_ERR_FLAG:
                continue
            ext = bool(cid & CAN_EFF_FLAG)
            frame = encode_frame(cid & (CAN_EFF_MASK if ext else CAN_SFF_MASK), data[:dlc], ext,
                                 bool(cid & CAN_RTR_FLAG))
            try:
                ser.write(frame)
            except serial.SerialException as e:
                print(f"Serial error: {e}")
                stop.set()
                break
            stats["to_bus"] += 1
            if args.verbose:
                print(f"  {args.iface} -> bus: {cid & CAN_EFF_MASK:08X} [{dlc}] {data[:dlc].hex(' ')}")

    threads = [threading.Thread(target=f, daemon=True) for f in (serial_to_can, can_to_serial)]
    for t in threads:
        t.start()
    try:
        last = time.monotonic()
        while not stop.is_set():
            time.sleep(0.2)
            if not args.verbose and time.monotonic() - last >= 5:
                last = time.monotonic()
                print(f"  frames: {stats['to_bus']} sent to bus, {stats['from_bus']} received from bus")
    except KeyboardInterrupt:
        pass
    stop.set()
    for t in threads:
        t.join(timeout=0.5)
    ser.close()
    sock.close()
    print("Bridge stopped.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--iface", default="vcan0")
    ap.add_argument("--bitrate", type=int, default=1_000_000, help="CAN bus bitrate (RobStride default 1 Mbps)")
    ap.add_argument("--mode", default="normal", choices=sorted(MODES))
    ap.add_argument("--serial-baud", type=int, default=2_000_000)
    ap.add_argument("--stopbits", type=int, default=2, choices=(1, 2))
    ap.add_argument("--selftest", action="store_true", help="adapter loopback test, then exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every frame")
    args = ap.parse_args()
    return selftest(args) if args.selftest else bridge(args)


if __name__ == "__main__":
    raise SystemExit(main())
