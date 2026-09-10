"""写错再涂改：count=0 身份、同 seed 稳定、不涂标点、窄框不丢对字。"""

from io import BytesIO

from PIL import Image

from renderer.core import RenderConfig, render
from renderer.layout import BlockSpec
from renderer.v3.mistakes import expand, expand_blocks, pick_wrong, scratch_skeleton
from renderer.v3.paper import CaptureSpec, PaperSpec
from renderer.v3.pipeline import (
    PageSpec,
    RenderRequest,
    _prepare_request,
    _wrap,
    _wrap_flagged,
    render as render_v3,
)

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


def _is_subsequence(needle: str, haystack: str) -> bool:
    cursor = iter(haystack)
    return all(char in cursor for char in needle)


def test_expand_zero_returns_same_object() -> None:
    text = "你好作业"
    out, marks = expand(text, 0, 11)
    assert out is text
    assert marks == frozenset()


def test_expand_no_eligible_returns_same_object() -> None:
    text = "Hello, world! 123。"
    out, marks = expand(text, 8, 11)
    assert out is text
    assert marks == frozenset()


def test_expand_same_seed_is_stable() -> None:
    text = "的是不了在有人这中大为上个国"
    a, ia = expand(text, 1, 42)
    b, ib = expand(text, 1, 42)
    assert a == b
    assert ia == ib
    assert a is not text
    assert len(a) == len(text) + 1
    assert ia


def test_lookalike_of_de() -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    assert pick_wrong("的", rng) == "地"


def test_wrap_flagged_matches_plain_wrap() -> None:
    text = "今天交 essay\n>>>落款：某某\n---\n第二页"
    plain = _wrap(text, 20, 0)
    flagged = _wrap_flagged(text, frozenset(), 20, 0)
    stripped = [([char for char, _flag in chars], align) for chars, align in flagged]
    assert stripped == plain


def test_count_zero_png_is_byte_identical() -> None:
    text = "你好作业"
    baseline = render(text, _v3())
    closed = render(text, _v3(scribble_count=0, scribble_style="strike"))
    assert baseline == closed


def test_count_one_changes_pixels() -> None:
    text = "的是不了在有人这中大"
    baseline = render(text, _v3(seed=9))
    marked = render(text, _v3(seed=9, scribble_count=1, scribble_style="cross"))
    assert marked != baseline
    assert marked[:8] == b"\x89PNG\r\n\x1a\n"
    Image.open(BytesIO(marked)).verify()

    result = render_v3(
        RenderRequest(
            text=text,
            page=_TINY,
            paper=PaperSpec(),
            capture=CaptureSpec(enabled=False),
            seed=9,
            max_pages=1,
            supersample=1,
            scribble_count=1,
            scribble_style="strike",
        )
    )
    assert result.images
    assert not result.metadata.get("missing_glyphs")


def test_narrow_block_keeps_original_chars() -> None:
    original = "你好作业"
    spec = BlockSpec(text=original, x=8, y=12, w=18, h=7, indent=0, glyph_scale=1.0)
    request = RenderRequest(
        text="",
        page=_TINY,
        paper=PaperSpec(ruling="blank"),
        capture=CaptureSpec(enabled=False),
        seed=3,
        layout_pages=[[spec]],
        scribble_count=8,
        scribble_style="scribble",
    )
    _prepare_request(request)
    new_text = request.layout_pages[0][0].text
    assert _is_subsequence(original, new_text)
    assert new_text.endswith("业")


def test_expand_blocks_respects_fits() -> None:
    original = ["你好世界"]
    texts, marks = expand_blocks(original, 3, 5, fits=lambda _i, _t: False)
    assert texts == original
    assert texts[0] is original[0]
    assert marks == [frozenset()]


def test_scratch_skeleton_three_styles() -> None:
    for style in ("strike", "scribble", "cross"):
        glyph = scratch_skeleton(style)
        assert glyph.char == f"\x00scratch:{style}"
        assert glyph.points.ndim == 3
