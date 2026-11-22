#!/usr/bin/env python3
"""Live keyboard-triggered drum strokes with collision-checked transfers.

Run from the openarm-dev repository root on macOS:
    mjpython projects/keyboard_teleop/drum_control_sim.py \
        --scene-directory runs/rl/music_preparation/scene \
        --profiles runs/rl/music_references/profiles

1-7 select drum; [ and ] change MIDI velocity; Space/R strike; Esc closes viewer.
Only the FIRST startup initializes MuJoCo. Subsequent strokes never reset qpos.
Uses existing TransferPlanner and RhythmicDrummingEnv.step_command(), not new PD gains.
This is a sequential (one stroke at a time) teleoperation prototype.
"""

import argparse
import collections
import json
import sys
import tempfile
import time
from pathlib import Path

import glfw
import mido
import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for relative in ("src", "projects/opendrummer_playground/src", "projects/cartesian_console/src",
                 "projects/drum_skill_learning/src"):
    directory = str(REPO_ROOT / relative)
    if directory not in sys.path:
        sys.path.insert(0, directory)

from openarm_rl.control.reference import SpeedConditionedReference
from openarm_rl.envs.rhythmic import RhythmicDrummingEnv
from openarm_rl.music.reference import TransferPlanner, load_strokes
from openarm_rl.music.score import ContactMatcher
from openarm_rl.paths import PROJECT_ROOT

DRUMS = (
    ("snare", 38), ("hi_hat", 42), ("floor_tom", 43),
    ("rack_tom_right", 47), ("rack_tom_left", 48),
    ("cymbal_left", 49), ("cymbal_right", 51),
)
SETTLE_SPEED_RAD_S = 0.08
TRANSFER_ERROR_RAD = 0.08


def single_note_midi(path, note, velocity, channel):
    """Only used for bootstrapping the existing drumming environment once."""
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("note_on", note=note, velocity=velocity, channel=channel, time=960))
    track.append(mido.Message("note_off", note=note, velocity=0, channel=channel, time=120))
    track.append(mido.MetaMessage("end_of_track", time=960))
    midi.save(str(path))


class DrumKeyboard:
    def __init__(self, scene, profiles, calibration, tempdir, channel):
        self.scene, self.profiles, self.calibration = scene, profiles, calibration
        self.selected, self.velocity, self.busy = 0, 80, False
        self.phase = "idle"
        self.last_result = "Choose drum; Space to strike"
        self.frame_clock = time.monotonic()
        self.frame_accumulator = 0.
        self.segments = collections.deque()
        self.active_segment = None
        self.segment_elapsed = 0.
        self.hit_count_at_start = 0
        self.chosen_drum = None
        self.chosen_arm = None

        bootstrap = tempdir / "bootstrap.mid"
        single_note_midi(bootstrap, DRUMS[0][1], 80, channel)
        self.env = RhythmicDrummingEnv(scene, profiles, bootstrap,
                                        score_only=True, velocity_calibration=calibration)
        self.env.reset(seed=800000)  # Exactly once: create a collision-free initial pose.
        # Live teleop has no predeclared MIDI target events. Keep hit detection and safety,
        # but avoid the bootstrap event being treated as an upcoming target.
        self.env.matcher = ContactMatcher([])
        # Keep the existing step_command controller alive between keyboard events.
        self.env.end = 1e6
        self.controller = self.env.controller
        self.hold_q = self.env.data.qpos[self.controller.q].copy()

        names = [name for name, _ in DRUMS]
        loaded, _, available = load_strokes(self.env.scene, profiles, names)
        self.banks = {key: SpeedConditionedReference(p["reference_bank"])
                      for key, p in loaded.items()}
        self.available = available
        self.planner = TransferPlanner.for_profiles(self.env.model, self.env.scene, profiles)
        self.velocity_calibration = self.env.velocity_calibration

    def _stroke_for(self, drum, arm):
        bank = self.banks[(drum, arm)]
        if self.velocity_calibration is None:
            speed = float(np.clip(.6, bank.speeds[0], bank.speeds[-1]))
        else:
            curve = self.velocity_calibration.curves[drum]
            midi_velocity = float(np.clip(self.velocity, curve[0, 1], curve[-1, 1]))
            speed = float(np.clip(self.velocity_calibration.speed(drum, midi_velocity),
                                  bank.speeds[0], bank.speeds[-1]))
        return bank.at(speed), speed

    def strike(self):
        if self.busy or self.env.unsafe:
            return
        drum, _ = DRUMS[self.selected]
        arms = self.available.get(drum, [])
        if not arms:
            self.last_result = f"No calibrated arm for {drum}"
            print(self.last_result, flush=True)
            return
        # Use the preferred arm encoded by the existing profile if possible.
        arm = arms[0]
        if len(arms) > 1:
            from openarm_rl.skills.registry import SKILLS
            preferred = SKILLS[drum].arm
            if preferred in arms:
                arm = preferred
        c = self.controller
        current = self.env.data.qpos[c.q].copy()
        speed_now = self.env.data.qvel[c.v].copy()
        if np.max(np.abs(speed_now)) > SETTLE_SPEED_RAD_S:
            self.last_result = "Arm still moving; try again when settled"
            print(self.last_result, flush=True)
            return

        try:
            stroke, impact_speed = self._stroke_for(drum, arm)
            active = slice(0, 7) if arm == "left" else slice(7, 14)
            ready = current.copy()
            ready[active] = stroke.positions[0, active]
            transfer = None
            if np.max(np.abs(ready-current)) > .001:
                transfer = self.planner.plan(current, ready, arm)
                # Planning checks the intended path, but executing from an unsettled
                # or substantially different actual pose would invalidate that check.
                if not np.isfinite(transfer.ends[-1]) or transfer.ends[-1] <= 0:
                    raise ValueError("Transfer planner returned invalid duration")
            segments = collections.deque()
            if transfer is not None:
                segments.append(("transfer", transfer, float(transfer.ends[-1]), arm))
            segments.append(("stroke", stroke, float(stroke.duration), arm))
        except (ValueError, KeyError) as exc:
            self.last_result = f"Planning rejected {drum}: {exc}"
            print(self.last_result, flush=True)
            return

        self.segments = segments
        self.active_segment = None
        self.segment_elapsed = 0.
        self.busy = True
        self.chosen_drum, self.chosen_arm = drum, arm
        self.hit_count_at_start = len(self.env.matcher.actual)
        self.frame_clock = time.monotonic()
        self.frame_accumulator = 0.
        self.last_result = f"{drum}/{arm}: target impact speed {impact_speed:.3f} m/s"
        print(self.last_result, flush=True)

    def keypress(self, key):
        if glfw.KEY_1 <= key <= glfw.KEY_7 and not self.busy:
            self.selected = key - glfw.KEY_1
            print("Selected", DRUMS[self.selected][0], flush=True)
        elif key == glfw.KEY_LEFT_BRACKET and not self.busy:
            self.velocity = max(1, self.velocity - 10)
        elif key == glfw.KEY_RIGHT_BRACKET and not self.busy:
            self.velocity = min(127, self.velocity + 10)
        elif key in (glfw.KEY_SPACE, glfw.KEY_R):
            self.strike()

    def _advance_segment(self):
        if self.segments:
            self.active_segment = self.segments.popleft()
            self.segment_elapsed = 0.
            self.phase = self.active_segment[0]
            return
        self.active_segment = None
        self.busy = False
        self.phase = "idle"
        self._summarize()
        self.hold_q = self.env.data.qpos[self.controller.q].copy()

    def _step_once(self):
        if self.env.unsafe:
            self.busy = False
            self.phase = "safety stop"
            self.last_result = f"SAFETY STOP: {self.env.failure}"
            print(self.last_result, flush=True)
            return
        if self.active_segment is None and self.busy:
            self._advance_segment()
        c = self.controller
        if self.active_segment is not None:
            kind, motion, duration, arm = self.active_segment
            t = min(self.segment_elapsed, duration)
            q, dq = motion.sample(t)
            # Stroke files contain a complete 14-joint reference based on their
            # calibration pose. The inactive arm must hold its *current* position.
            selected = slice(0, 7) if arm == "left" else slice(7, 14)
            desired = self.hold_q.copy()
            desired[selected] = q[selected]
            velocity = np.zeros(14)
            velocity[selected] = dq[selected]
            measured, _ = self.env.encoders()
            motor = self.env.servo.position_target(desired, measured)
            if kind == "stroke" and len(motion.active_joints) == 1:
                # Preserve wrist compliance for legacy wrist-only strokes.
                adjusted = motion.motor_target(t, measured, self.env.servo)
                motor[motion.wrist] = adjusted[motion.wrist]
            motor = np.clip(motor, c.bounds[:, 0], c.bounds[:, 1])
            velocity = np.clip(velocity, -c.velocity, c.velocity)
            self.hold_q = desired.copy()
        else:
            desired = self.hold_q.copy()
            velocity = np.zeros(14)
            measured, _ = self.env.encoders()
            motor = np.clip(self.env.servo.position_target(desired, measured),
                            c.bounds[:, 0], c.bounds[:, 1])
        try:
            self.env.step_command(motor, velocity, target_position=desired)
        except (ValueError, RuntimeError) as exc:
            self.busy = False
            self.phase = "error"
            self.last_result = f"Execution stopped: {exc}"
            self.active_segment = None
            self.segments.clear()
            print(self.last_result, flush=True)
            return
        # This environment logs traces for RL. Don't grow memory indefinitely in teleop.
        if len(self.env.trace) > 1000:
            del self.env.trace[:-1000]
        if self.active_segment is not None:
            self.segment_elapsed += self.env.period
            if self.segment_elapsed >= self.active_segment[2] - self.env.period/2:
                kind, motion, duration, arm = self.active_segment
                if kind == "transfer":
                    selected = slice(0, 7) if arm == "left" else slice(7, 14)
                    final_q, _ = motion.sample(duration)
                    actual = self.env.data.qpos[c.q]
                    error = float(np.max(np.abs(actual[selected]-final_q[selected])))
                    if error > TRANSFER_ERROR_RAD:
                        self.busy = False
                        self.phase = "transfer error"
                        self.last_result = f"Transfer tracking error {error:.3f} rad; strike cancelled"
                        print(self.last_result, flush=True)
                        self.segments.clear()
                        self.active_segment = None
                        self.hold_q = actual.copy()
                        return
                self._advance_segment()

    def tick(self):
        now = time.monotonic()
        dt = min(now-self.frame_clock, .05)
        self.frame_clock = now
        self.frame_accumulator += max(0, dt)
        while self.frame_accumulator >= self.env.period:
            self.frame_accumulator -= self.env.period
            self._step_once()

    def _summarize(self):
        hits = self.env.matcher.actual[self.hit_count_at_start:]
        if hits:
            primary = next((hit for hit in hits if hit["target"] == self.chosen_drum), hits[0])
            speed = primary.get("downward_speed_m_s")
            self.last_result = (f"{primary['target']}: {speed:.3f} m/s" if speed is not None
                                else f"{primary['target']}: contact recorded")
            if primary.get("midi_velocity") is not None:
                self.last_result += f", MIDI {primary['midi_velocity']:.1f}"
            self.last_result += f"; hits={len(hits)}"
        else:
            self.last_result = f"{self.chosen_drum}: no contact detected"
        print(self.last_result, flush=True)

    def overlay(self):
        labels = "Drum\nRequested MIDI velocity\nCalibration\nPhase\nLast result"
        values = (f"{DRUMS[self.selected][0]}\n{self.velocity}\n"
                  f"{'enabled' if self.calibration else 'nominal only'}\n"
                  f"{self.phase}\n{self.last_result}")
        return labels, values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-directory", type=Path, default=Path("runs/rl/music_preparation/scene"))
    parser.add_argument("--profiles", type=Path, default=Path("runs/rl/music_references/profiles"))
    parser.add_argument("--velocity-calibration", type=Path)
    args = parser.parse_args(argv)
    if sys.platform == "darwin" and getattr(mujoco.viewer, "_MJPYTHON", None) is None:
        parser.error("On macOS run this with mjpython")
    for filepath in (args.scene_directory / "scene.json", args.scene_directory / "model.mjb"):
        if not filepath.is_file():
            parser.error(f"Missing scene asset: {filepath}")
    for drum, _ in DRUMS:
        if not any((args.profiles / f"{drum}{suffix}.json").is_file()
                   for suffix in ("", ".left", ".right")):
            parser.error(f"Missing reference profile for {drum}")
    if args.velocity_calibration and not args.velocity_calibration.is_file():
        parser.error(f"Missing calibration: {args.velocity_calibration}")

    mapping = json.loads((PROJECT_ROOT / "configs/music/percussion.json").read_text())
    with tempfile.TemporaryDirectory(prefix="openarm_live_drum_") as directory:
        teleop = DrumKeyboard(args.scene_directory, args.profiles, args.velocity_calibration,
                              Path(directory), mapping["channel"])
        events = collections.deque()
        def on_key(key):
            events.append(key)
        print("1-7: select drum | [ / ]: MIDI velocity | Space/R: strike", flush=True)
        print("Continuous MuJoCo state. Transfers are planned and executed before strokes.", flush=True)
        with mujoco.viewer.launch_passive(
            teleop.env.model,
            teleop.env.data,
            key_callback=on_key
        ) as viewer:
            geomgroup = viewer.opt.geomgroup.copy()
            shown = None

            while viewer.is_running():
                with viewer.lock():
                    while events:
                        teleop.keypress(events.popleft())

                    # Prevent number keys from hiding MuJoCo geometry.
                    viewer.opt.geomgroup[:] = geomgroup

                    teleop.tick()
                    overlay = teleop.overlay()
                viewer.sync()
                if overlay != shown:
                    viewer.set_texts([
                        (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                         "1-7 / [ ] / Space", "drum / intensity / strike"),
                        (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, *overlay),
                    ])
                    shown = overlay
                time.sleep(.003)
        teleop.env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
