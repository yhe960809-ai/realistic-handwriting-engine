"""v3 笔具与墨色：默认不换笔、换色/换笔真的改变输出。"""

from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from renderer.core import RenderConfig, render
from renderer.v3.paper import CaptureSpec, PaperSpec
from renderer.v3.pipeline import PageSpec, RenderRequest, render as render_v3
from renderer.v3.style import PENS

_TINY = PageSpec(
    width_mm=70.0,
    height_mm=40.0,
    dpi=120.0,
    margin_left_mm=4.0,
    margin_right_mm=4.0,
    margin_top_mm=4.0,
    margin_bottom_mm=4.0,
    glyph_size_mm=7.0,
    line_pitch_mm=10.0,
    indent=0,
    bind_to_ruling=False,
)


def _v3(**overrides) -> RenderConfig:
    base = dict(
        engine="v3",
        mode="preview",
        max_pages=1,
        seed=7,
        page_width=500,
        page_height=360,
        margin=24,
        font_size=36,
        line_spacing=56,
        paper_style="plain",
        filter_style="none",
        paragraph_indent=0,
        antialias="off",
    )
    base.update(overrides)
    return RenderConfig(**base)


def test_default_black_hex_is_noop() -> None:
    text = "你好作业"
    baseline = render(text, _v3())
    matched = render(text, _v3(pen_color="#1a203a"))
    assert baseline == matched


def test_red_ink_changes_pixels() -> None:
    text = "你好作业"
    baseline = render(text, _v3())
    red = render(text, _v3(pen_color="#8f1f1f"))
    assert red != baseline
    arr = np.asarray(Image.open(BytesIO(red)).convert("RGB"))
    ink = arr.min(axis=2) < 180
    assert ink.any()
    samples = arr[ink]
    assert float(samples[:, 0].mean()) > float(samples[:, 2].mean())


def test_blue_ink_changes_pixels() -> None:
    text = "你好作业"
    baseline = render(text, _v3())
    blue = render(text, _v3(pen_color="#1e3a70"))
    assert blue != baseline


def test_thin_pen_differs_from_thick() -> None:
    text = "你好作业"
    thin = render(text, _v3(pen_id="gel-0.38"))
    thick = render(text, _v3(pen_id="ballpoint-0.7"))
    assert thin != thick


def test_fountain_pen_metadata_stays_when_medium() -> None:
    result = render_v3(
        RenderRequest(
            text="钢笔正楷",
            style_id="fountain-formal",
            page=_TINY,
            paper=PaperSpec(),
            capture=CaptureSpec(enabled=False),
            seed=7,
            max_pages=1,
            supersample=1,
        )
    )
    assert result.metadata["pen_id"] == "fountain-f"


def test_pen_override_changes_metadata() -> None:
    result = render_v3(
        RenderRequest(
            text="换笔",
            style_id="fountain-formal",
            page=_TINY,
            paper=PaperSpec(),
            capture=CaptureSpec(enabled=False),
            seed=7,
            max_pages=1,
            supersample=1,
            pen=PENS["gel-0.38"],
        )
    )
    assert result.metadata["pen_id"] == "gel-0.38"


def test_invalid_pen_id_raises() -> None:
    with pytest.raises(ValueError, match="pen_id"):
        render("你好", _v3(pen_id="marker-9"))
