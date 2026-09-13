"""LossCurve is the boundary between the two implementations, so it round-trips exactly."""

from __future__ import annotations

import math

from swarm_mlp.utils.curves import LossCurve


def _sample_curve() -> LossCurve:
    curve = LossCurve(meta={"source": "test", "seed": 0, "learning_rate": 1e-3})
    for step in range(1, 6):
        curve.record(step * 64, math.pi / step, 1.0 - 1.0 / (step + 1))
    return curve


def test_round_trip_is_exact(tmp_path):
    original = _sample_curve()
    reloaded = LossCurve.load(original.save(tmp_path / "curve.json"))

    # `==`, not `allclose`: rounding here would silently cap the precision of
    # every comparison built on top of this file.
    assert reloaded.samples == original.samples
    assert reloaded.loss == original.loss
    assert reloaded.accuracy == original.accuracy
    assert len(reloaded) == len(original)


def test_meta_round_trips(tmp_path):
    original = _sample_curve()
    reloaded = LossCurve.load(original.save(tmp_path / "curve.json"))
    assert reloaded.meta == original.meta


def test_save_creates_parent_directories(tmp_path):
    target = tmp_path / "deeply" / "nested" / "curve.json"
    assert _sample_curve().save(target) == target
    assert target.is_file()


def test_empty_curve_round_trips(tmp_path):
    reloaded = LossCurve.load(LossCurve().save(tmp_path / "empty.json"))
    assert len(reloaded) == 0
    assert reloaded.meta == {}
