# robstride_can

Low-level SocketCAN driver for RobStride motors (RS00–RS06), using only the Python standard library (plus pyserial for the Waveshare bridge).

| File | What it does |
|---|---|
| `robstride.py` | The driver. It speaks RobStride's private protocol (29-bit extended IDs) on a raw CAN socket. The main call is `bus.mit(id, p, v, kp, kd, t)`. |
| `test_sweep.py` | Slowly sweeps one motor +90° from its current position and back, for a few cycles. |
| `scan.py` | Pings CAN ids to find motors. Nothing moves. |
| `waveshare_bridge.py` | Bridges a Waveshare USB-CAN-A (serial) adapter to a virtual SocketCAN interface (`vcan0`). |
| `setup_vcan.sh` | Creates `vcan0` (needs sudo, once per boot). |

## Using the Waveshare USB-CAN-A

The USB-CAN-A shows up as `/dev/ttyUSB0` (CH340 chip), not as `can0`, because it uses Waveshare's own serial protocol. `waveshare_bridge.py` turns it into the virtual interface `vcan0`, and everything else just uses `--channel vcan0`.

### One-time setup

```bash
sudo usermod -aG dialout $USER     # lets you open /dev/ttyUSB0; log out and back in afterwards
```

### Every boot

```bash
cd projects/robstride_can
./setup_vcan.sh                    # creates vcan0
```

### Test step by step

1. **Adapter check (no motor needed):** `python3 waveshare_bridge.py --selftest`
   This puts the adapter in loopback mode and checks that frames come back. If it fails, try `--stopbits 1` or replug the adapter.
2. **Start the bridge** in its own terminal and leave it running:
   `python3 waveshare_bridge.py` (add `-v` to print every frame)
3. **Find the motor** (it won't move): `python3 scan.py --channel vcan0`
4. **Sweep:** `python3 test_sweep.py --channel vcan0 --id 1 --model RS03`

Useful sweep flags: `--amplitude -90` (sweep the other way), `--cycles 5`, `--period 8` (slower), `--kp`, `--kd`, `--host-id`.

### Wiring checklist

- Connect CAN_H to CAN_H and CAN_L to CAN_L, and connect ground between the adapter and the motor if possible.
- Terminate the bus with 120 Ω at both ends. The USB-CAN-A has a termination switch or jumper; turn it on if the adapter sits at one end of the bus.
- Power the motor (check the voltage range on its datasheet).
- RobStride motors default to 1 Mbps (`--bitrate` on the bridge must match). Factory CAN id is often 127, so scan the full range (`scan.py` does 1-127 by default).

## Native SocketCAN adapters

With an adapter that the kernel supports directly (candleLight/gs_usb, PEAK, MCP2515 HAT), skip the bridge:

```bash
sudo ip link set can0 up type can bitrate 1000000
python3 test_sweep.py --channel can0 --id 1 --model RS03
```

## Driver API

```python
from robstride import RobStrideBus

with RobStrideBus("vcan0") as bus:
    bus.add_motor(1, "RS03")      # model picks the scaling ranges
    bus.enable(1)                 # stop + clear faults, run_mode=0 (MIT), enable
    fb = bus.mit(1, p=0.0, v=0.0, kp=20.0, kd=1.0, t=0.0)
    print(fb.position, fb.velocity, fb.torque, fb.temperature, fb.faults)
    bus.disable(1)
```

Also available: `ping`, `stop`/`disable`, `set_zero`, `read_param`, `write_param` (see `Param` for indices), and `feedback(id)` for the latest cached state.

Notes:
- The motor only moves while it receives commands. Send `mit()` continuously, around 100–500 Hz.
- MIT positions are limited to ±4π rad around the motor's mechanical zero.
- `disable()` makes the motor go limp.
