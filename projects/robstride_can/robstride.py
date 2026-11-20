"""Low-level SocketCAN driver for RobStride motors (RS00-RS06).

Speaks RobStride's *private* CAN protocol (29-bit extended IDs, 1 Mbps default)
directly over a Linux raw CAN socket. No third-party dependencies.

Extended-ID layout (29 bits)::

    bits 28..24  communication type
    bits 23..8   "data area 2" (meaning depends on type; usually host id in 15..8)
    bits  7..0   destination (motor id when host->motor, host id when motor->host)

The main API is :meth:`RobStrideBus.mit` which sends a type-1 "motion control"
frame -- RobStride's MIT-style impedance command::

    torque = kp * (p_des - p) + kd * (v_des - v) + t_ff

Every command (and enable/stop) is answered by a type-2 feedback frame, which a
background RX thread decodes into :class:`Feedback` per motor.

Typical use::

    with RobStrideBus("can0") as bus:
        bus.add_motor(1, "RS03")
        bus.enable(1)                       # stops, sets run_mode=0 (MIT), enables
        fb = bus.mit(1, p=0.0, v=0.0, kp=20.0, kd=1.0, t=0.0)
        print(fb.position, fb.velocity, fb.torque)
        bus.disable(1)
"""

from __future__ import annotations

import math
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------


class CommType:
    GET_DEVICE_ID = 0
    MOTION_CONTROL = 1  # MIT-style command
    FEEDBACK = 2
    ENABLE = 3
    STOP = 4
    SET_ZERO = 6
    SET_CAN_ID = 7
    READ_PARAM = 17
    WRITE_PARAM = 18
    FAULT_FEEDBACK = 21
    SET_BAUD = 22
    ACTIVE_REPORT = 24


class Param:
    """Single-parameter indices (read with type 17, write with type 18)."""

    RUN_MODE = 0x7005  # uint8: 0 MIT/operation, 1 PP, 2 speed, 3 current, 5 CSP
    IQ_REF = 0x7006
    SPD_REF = 0x700A
    LIMIT_TORQUE = 0x700B
    CUR_KP = 0x7010
    CUR_KI = 0x7011
    CUR_FILT_GAIN = 0x7014
    LOC_REF = 0x7016
    LIMIT_SPD = 0x7017
    LIMIT_CUR = 0x7018
    MECH_POS = 0x7019
    IQF = 0x701A
    MECH_VEL = 0x701B
    VBUS = 0x701C
    LOC_KP = 0x701E
    SPD_KP = 0x701F
    SPD_KI = 0x7020
    SPD_FILT_GAIN = 0x7021


# Parameters stored as uint8 rather than float32.
_UINT8_PARAMS = {Param.RUN_MODE}


class RunMode:
    MIT = 0
    POSITION_PP = 1
    SPEED = 2
    CURRENT = 3
    POSITION_CSP = 5


@dataclass(frozen=True)
class MotorLimits:
    """Ranges used to scale the 16-bit fields of type-1/type-2 frames."""

    p_max: float  # rad   (symmetric, -p_max..p_max)
    v_max: float  # rad/s (symmetric)
    t_max: float  # Nm    (symmetric)
    kp_max: float  # 0..kp_max
    kd_max: float  # 0..kd_max


MODEL_LIMITS: dict[str, MotorLimits] = {
    "RS00": MotorLimits(4 * math.pi, 33.0, 14.0, 500.0, 5.0),
    "RS01": MotorLimits(4 * math.pi, 44.0, 17.0, 500.0, 5.0),
    "RS02": MotorLimits(4 * math.pi, 44.0, 17.0, 500.0, 5.0),
    "RS03": MotorLimits(4 * math.pi, 50.0, 60.0, 5000.0, 100.0),
    "RS04": MotorLimits(4 * math.pi, 15.0, 120.0, 5000.0, 100.0),
    "RS05": MotorLimits(4 * math.pi, 50.0, 5.5, 500.0, 5.0),
    "RS06": MotorLimits(4 * math.pi, 50.0, 36.0, 5000.0, 100.0),
}

FAULT_BITS = {
    0: "undervoltage",
    1: "driver fault / overcurrent",
    2: "over-temperature",
    3: "magnetic encoder fault",
    4: "stall overload",
    5: "uncalibrated",
}

MODE_NAMES = {0: "reset", 1: "calibration", 2: "run"}

# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------


def float_to_uint(x: float, x_min: float, x_max: float, bits: int = 16) -> int:
    x = min(max(x, x_min), x_max)
    span = x_max - x_min
    return int(round((x - x_min) * ((1 << bits) - 1) / span))


def uint_to_float(u: int, x_min: float, x_max: float, bits: int = 16) -> float:
    return u * (x_max - x_min) / ((1 << bits) - 1) + x_min


def make_id(comm_type: int, data2: int, dest: int) -> int:
    return ((comm_type & 0x1F) << 24) | ((data2 & 0xFFFF) << 8) | (dest & 0xFF)


def split_id(can_id: int) -> tuple[int, int, int]:
    return (can_id >> 24) & 0x1F, (can_id >> 8) & 0xFFFF, can_id & 0xFF


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


@dataclass
class Feedback:
    motor_id: int
    position: float  # rad
    velocity: float  # rad/s
    torque: float  # Nm
    temperature: float  # deg C
    mode: int  # 0 reset, 1 calibration, 2 run
    fault_bits: int
    stamp: float = field(default_factory=time.monotonic)

    @property
    def mode_name(self) -> str:
        return MODE_NAMES.get(self.mode, f"unknown({self.mode})")

    @property
    def faults(self) -> list[str]:
        return [name for bit, name in FAULT_BITS.items() if self.fault_bits & (1 << bit)]


class RobStrideError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# SocketCAN transport
# ---------------------------------------------------------------------------

_CAN_FRAME = struct.Struct("=IB3x8s")  # struct can_frame
_CAN_EFF_FLAG = 0x80000000
_CAN_RTR_FLAG = 0x40000000
_CAN_ERR_FLAG = 0x20000000
_CAN_EFF_MASK = 0x1FFFFFFF


class _SocketCAN:
    def __init__(self, channel: str):
        self.sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        # Only accept extended data frames (RobStride private protocol is all EFF).
        # Don't put CAN_ERR_FLAG in the mask: the kernel then files the filter as an
        # error-frame filter and no data frames are delivered at all.
        flt = struct.pack("=II", _CAN_EFF_FLAG, _CAN_EFF_FLAG | _CAN_RTR_FLAG)
        self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, flt)
        self.sock.bind((channel,))
        self.sock.settimeout(0.05)

    def send(self, can_id: int, data: bytes) -> None:
        data = data.ljust(8, b"\x00")
        self.sock.send(_CAN_FRAME.pack(can_id | _CAN_EFF_FLAG, 8, data))

    def recv(self) -> tuple[int, bytes] | None:
        try:
            raw = self.sock.recv(16)
        except socket.timeout:
            return None
        can_id, dlc, data = _CAN_FRAME.unpack(raw)
        return can_id & _CAN_EFF_MASK, data[:dlc]

    def close(self) -> None:
        self.sock.close()


# ---------------------------------------------------------------------------
# Bus / driver
# ---------------------------------------------------------------------------


class RobStrideBus:
    """One CAN interface with any number of RobStride motors on it."""

    def __init__(self, channel: str = "can0", host_id: int = 0xFD, reply_timeout: float = 0.05):
        self.channel = channel
        self.host_id = host_id & 0xFF
        self.reply_timeout = reply_timeout
        self._limits: dict[int, MotorLimits] = {}
        self._feedback: dict[int, Feedback] = {}
        self._fb_events: dict[int, threading.Event] = {}
        self._param_replies: queue.Queue[tuple[int, int, bytes]] = queue.Queue()
        self._lock = threading.Lock()
        self._tx_lock = threading.Lock()
        self._can = _SocketCAN(channel)
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, name="robstride-rx", daemon=True)
        self._rx_thread.start()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._running = False
        self._rx_thread.join(timeout=0.5)
        self._can.close()

    def __enter__(self) -> "RobStrideBus":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def add_motor(self, motor_id: int, model: str | MotorLimits = "RS03") -> None:
        limits = model if isinstance(model, MotorLimits) else MODEL_LIMITS[model.upper()]
        with self._lock:
            self._limits[motor_id] = limits
            self._fb_events.setdefault(motor_id, threading.Event())

    def limits(self, motor_id: int) -> MotorLimits:
        try:
            return self._limits[motor_id]
        except KeyError:
            raise RobStrideError(f"motor {motor_id} not registered; call add_motor() first") from None

    # -- RX ----------------------------------------------------------------

    def _rx_loop(self) -> None:
        while self._running:
            try:
                frame = self._can.recv()
            except OSError:
                if self._running:
                    time.sleep(0.01)
                continue
            if frame is None:
                continue
            can_id, data = frame
            comm_type, data2, _dest = split_id(can_id)
            motor_id = data2 & 0xFF
            if comm_type == CommType.FEEDBACK and len(data) == 8:
                self._handle_feedback(motor_id, data2, data)
            elif comm_type == CommType.READ_PARAM:
                self._param_replies.put((motor_id, int.from_bytes(data[0:2], "little"), data))

    def _handle_feedback(self, motor_id: int, data2: int, data: bytes) -> None:
        lim = self._limits.get(motor_id)
        if lim is None:
            return
        p_u, v_u, t_u, temp_u = struct.unpack(">HHHH", data)
        fb = Feedback(
            motor_id=motor_id,
            position=uint_to_float(p_u, -lim.p_max, lim.p_max),
            velocity=uint_to_float(v_u, -lim.v_max, lim.v_max),
            torque=uint_to_float(t_u, -lim.t_max, lim.t_max),
            temperature=temp_u / 10.0,
            fault_bits=(data2 >> 8) & 0x3F,
            mode=(data2 >> 14) & 0x03,
        )
        with self._lock:
            self._feedback[motor_id] = fb
            ev = self._fb_events.get(motor_id)
        if ev is not None:
            ev.set()

    # -- TX helpers ----------------------------------------------------------

    def _send(self, comm_type: int, data2: int, motor_id: int, data: bytes = b"") -> None:
        with self._tx_lock:
            self._can.send(make_id(comm_type, data2, motor_id), data)

    def _send_and_wait_feedback(
        self, comm_type: int, data2: int, motor_id: int, data: bytes = b"", timeout: float | None = None
    ) -> Feedback | None:
        ev = self._fb_events.setdefault(motor_id, threading.Event())
        ev.clear()
        self._send(comm_type, data2, motor_id, data)
        if timeout is None:
            timeout = self.reply_timeout
        if timeout > 0 and ev.wait(timeout):
            return self.feedback(motor_id)
        return None

    def feedback(self, motor_id: int) -> Feedback | None:
        """Latest feedback received from ``motor_id`` (or None)."""
        with self._lock:
            return self._feedback.get(motor_id)

    # -- public commands -----------------------------------------------------

    def mit(
        self,
        motor_id: int,
        p: float = 0.0,
        v: float = 0.0,
        kp: float = 0.0,
        kd: float = 0.0,
        t: float = 0.0,
        wait: bool = True,
    ) -> Feedback | None:
        """Send one MIT/motion-control frame (comm type 1).

        Values are clamped to the motor model's ranges. Returns the feedback
        frame the motor replies with (None if ``wait`` is False or it timed out).
        Motor must be enabled in run_mode 0 (see :meth:`enable`).
        """
        lim = self.limits(motor_id)
        t_u = float_to_uint(t, -lim.t_max, lim.t_max)
        payload = struct.pack(
            ">HHHH",
            float_to_uint(p, -lim.p_max, lim.p_max),
            float_to_uint(v, -lim.v_max, lim.v_max),
            float_to_uint(kp, 0.0, lim.kp_max),
            float_to_uint(kd, 0.0, lim.kd_max),
        )
        if not wait:
            self._send(CommType.MOTION_CONTROL, t_u, motor_id, payload)
            return None
        return self._send_and_wait_feedback(CommType.MOTION_CONTROL, t_u, motor_id, payload)

    def enable(self, motor_id: int, run_mode: int = RunMode.MIT, clear_faults: bool = True) -> Feedback:
        """Stop, (optionally) clear faults, set run_mode, then enable."""
        self.stop(motor_id, clear_faults=clear_faults)
        self.write_param(motor_id, Param.RUN_MODE, run_mode)
        time.sleep(0.01)
        fb = self._send_and_wait_feedback(CommType.ENABLE, self.host_id, motor_id, timeout=0.2)
        if fb is None:
            raise RobStrideError(f"motor {motor_id}: no reply to enable on {self.channel}")
        if fb.faults:
            raise RobStrideError(f"motor {motor_id}: faults after enable: {fb.faults}")
        return fb

    def stop(self, motor_id: int, clear_faults: bool = False) -> Feedback | None:
        """Disable the motor (comm type 4). Motor goes limp."""
        data = b"\x01" if clear_faults else b""
        return self._send_and_wait_feedback(CommType.STOP, self.host_id, motor_id, data, timeout=0.1)

    disable = stop

    def set_zero(self, motor_id: int) -> Feedback | None:
        """Set current position as mechanical zero (comm type 6). Not persisted until saved."""
        return self._send_and_wait_feedback(CommType.SET_ZERO, self.host_id, motor_id, b"\x01", timeout=0.1)

    def ping(self, motor_id: int, timeout: float = 0.1) -> Feedback | None:
        """Probe a motor without moving it (a disable/stop frame elicits feedback)."""
        return self._send_and_wait_feedback(CommType.STOP, self.host_id, motor_id, timeout=timeout)

    def write_param(self, motor_id: int, index: int, value: float | int) -> None:
        if index in _UINT8_PARAMS:
            val = struct.pack("<B3x", int(value))
        else:
            val = struct.pack("<f", float(value))
        data = struct.pack("<HH", index, 0) + val
        self._send_and_wait_feedback(CommType.WRITE_PARAM, self.host_id, motor_id, data, timeout=0.05)

    def read_param(self, motor_id: int, index: int, timeout: float = 0.1) -> float | int:
        while not self._param_replies.empty():  # drop stale replies
            self._param_replies.get_nowait()
        self._send(CommType.READ_PARAM, self.host_id, motor_id, struct.pack("<HH4x", index, 0))
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                mid, idx, data = self._param_replies.get(timeout=remaining)
            except queue.Empty:
                break
            if mid == motor_id and idx == index:
                if index in _UINT8_PARAMS:
                    return data[4]
                return struct.unpack("<f", data[4:8])[0]
        raise RobStrideError(f"motor {motor_id}: no reply reading param 0x{index:04X}")
