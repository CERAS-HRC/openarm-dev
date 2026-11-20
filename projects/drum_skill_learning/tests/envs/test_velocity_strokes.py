import json
from pathlib import Path

import mido
import pytest

from openarm_rl.control.reference import JointReference
from openarm_rl.envs.rhythmic import RhythmicDrummingEnv
from openarm_rl.music.preflight import inspect_performance
from openarm_rl.planning.rhythmic_references import compact_reference


@pytest.fixture
def velocity_score(rhythm_options, tmp_path):
    profile = json.loads((Path(rhythm_options["profiles"]) / "snare.json").read_text())
    stroke = JointReference.load(profile["reference_bank"]["references"][0])
    # Two distinguishable strokes sharing one ready pose: the bank's soft and loud ends.
    soft = compact_reference(stroke, approach=.08, retract=.10, settle=.03).document()
    loud = compact_reference(stroke, approach=.03, retract=.05, settle=.01).document()
    profile["reference_bank"] = {"schema": "openarm.speed_conditioned_strike.v1",
                                 "speeds_m_s": [.35, .85], "references": [soft, loud]}
    (tmp_path / "snare.json").write_text(json.dumps(profile))
    calibration = {"schema": "openarm.drum_velocity_calibration.v1", "measurement_source": "test",
                   "drums": {"snare": [[.3, 1], [.6, 64], [.9, 127]]}}
    (tmp_path / "calibration.json").write_text(json.dumps(calibration))
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend([mido.Message("note_on", channel=9, note=38, velocity=10),
                  mido.Message("note_off", channel=9, note=38, velocity=0, time=1),
                  mido.Message("note_on", channel=9, note=38, velocity=120, time=1919),
                  mido.Message("note_off", channel=9, note=38, velocity=0, time=1)])
    midi.save(tmp_path / "score.mid")
    return {"scene_directory": rhythm_options["scene_directory"], "profiles": tmp_path,
            "midi_file": tmp_path / "score.mid", "calibration": tmp_path / "calibration.json"}


def test_note_velocity_selects_bank_stroke(velocity_score):
    options = {key: velocity_score[key] for key in ("scene_directory", "profiles", "midi_file")}
    with RhythmicDrummingEnv(**options, velocity_calibration=velocity_score["calibration"]) as env:
        reference = env.reference
        soft, loud = sorted(reference.strokes, key=lambda stroke: stroke["goal"].time_s)
        # Velocity 10 inverts to ~.34 m/s and 120 to ~.87 m/s; both clamp to the bank's .35-.85 range.
        assert reference.stroke_speeds[soft["goal"].event_id] == pytest.approx(.35)
        assert reference.stroke_speeds[loud["goal"].event_id] == pytest.approx(.85)
        assert soft["reference"].duration > loud["reference"].duration
        assert reference.document()["event_stroke_speeds_m_s"] == reference.stroke_speeds


def test_without_calibration_every_note_uses_the_nominal_stroke(velocity_score):
    options = {key: velocity_score[key] for key in ("scene_directory", "profiles", "midi_file")}
    with RhythmicDrummingEnv(**options) as env:
        first, second = env.reference.strokes
        assert first["reference"] is second["reference"]
        assert env.reference.stroke_speeds == {}


def test_preflight_schedules_the_velocity_selected_strokes(velocity_score):
    report = inspect_performance(velocity_score["midi_file"], velocity_score["scene_directory"],
                                 velocity_score["profiles"], velocity_calibration=velocity_score["calibration"])
    assert report["executable"], report["issues"]


def test_measured_bank_below_nominal_speed_plays_distinct_hits(rhythm_options, tmp_path, monkeypatch):
    from openarm_rl.music.corpus import write_segment
    from openarm_rl.planning import rhythmic_references

    # Probe three downstroke timings on the same physical snare pose.
    monkeypatch.setattr(rhythmic_references, "STROKE_PROBES",
                        [(.03, .05, .01, scale) for scale in (.65, 1., 1.6)])
    scene = Path(rhythm_options["scene_directory"])
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    rhythmic_references.calibrate(scene, Path(rhythm_options["profiles"]) / "snare.json",
                                  tmp_path, .5, 2., 8.)
    bank = json.loads((profiles / "snare.json").read_text())["reference_bank"]
    soft, middle, loud = bank["speeds_m_s"]
    assert bank["calibration_speed_m_s"] == [soft, middle, loud]
    assert 0 < soft < middle < loud < .6

    curve = tmp_path / "sim-scale.json"
    curve.write_text(json.dumps({"schema": "openarm.drum_velocity_calibration.v1",
                                 "measurement_source": "simulation test scale",
                                 "drums": {"snare": [[soft, 1], [middle, 64], [loud, 127]]}}))
    midi = tmp_path / "sweep.mid"
    write_segment(midi, [{"time_s": .5 + i, "note": 38, "velocity": v}
                         for i, v in enumerate((1, 64, 127))], 0., 1.)
    with RhythmicDrummingEnv(scene, profiles, midi, velocity_calibration=curve, score_only=True) as env:
        env.reset(seed=2026)
        while env.active:
            _, _, _, _, info = env.step(env.reference_action())
    assert info["is_success"] and info["correct_hits"] == 3 and info["extra_hits"] == 0
    measured = [hit["downward_speed_m_s"] for hit in info["hits"]]
    assert measured == pytest.approx([soft, middle, loud], abs=.025)
