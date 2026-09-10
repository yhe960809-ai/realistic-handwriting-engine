"""出锋顿笔与纸纤维吸水：只测物理层，不渲整页。"""

from dataclasses import replace

import numpy as np

from renderer.v3.ink import apply_absorbency_bleed
from renderer.v3.strokes import PIE, profile_for
from renderer.v3.style import get_style
from renderer.v3.supplement import get as get_supplement
from renderer.v3.trajectory import synthesize


def _pie_tail_width(exit_pressed: float) -> float:
    assert profile_for(PIE).tapers_out
    skeleton = get_supplement("，")
    assert skeleton is not None
    base = get_style("xingkai-daily")
    style = replace(base, brush=replace(base.brush, exit_pressed=exit_pressed))
    paths = synthesize(skeleton, style, seed=7, occurrence=0, em_px=40.0)
    path = next(item for item in paths if not item.is_ligature)
    # 出锋在弧长末端最强：exit_pressed=0 乘 taper≈0.06，=1 保持剖面宽度。
    return float(path.width_scale[-1])


def test_exit_pressed_thickens_taper_tip():
    sharp = _pie_tail_width(0.0)
    pressed = _pie_tail_width(1.0)
    # 尖收会被 width_scale 下限 0.03 钳住；顿笔应明显高于这条地板。
    assert pressed > sharp
    assert pressed >= 0.06 - 1e-12


def test_legacy_exit_pressed_is_noop(monkeypatch):
    monkeypatch.setattr("renderer.v3.trajectory.USE_LEGACY_INK", True)
    sharp = _pie_tail_width(0.0)
    pressed = _pie_tail_width(1.0)
    assert abs(pressed - sharp) < 1e-6


def test_absorbency_neutral_is_identity():
    mass = np.zeros((48, 48), dtype=np.float32)
    mass[16:32, 16:32] = 1.0
    absorbency = np.full_like(mass, 0.5)
    out = apply_absorbency_bleed(mass, absorbency, sigma=1.4)
    np.testing.assert_allclose(out, mass, atol=1e-6)


def test_absorbency_high_spreads_more_than_low():
    mass = np.zeros((48, 48), dtype=np.float32)
    mass[20:28, 20:28] = 1.0
    high = apply_absorbency_bleed(mass, np.full_like(mass, 0.9), sigma=1.4)
    low = apply_absorbency_bleed(mass, np.full_like(mass, 0.1), sigma=1.4)
    assert float(high[32, 24]) > float(low[32, 24])
    assert float(high[24, 24]) < float(low[24, 24])
