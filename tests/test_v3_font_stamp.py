"""生僻汉字字体盖章：不误伤原路线、有墨、无豆腐、同 seed 一致。"""

from io import BytesIO

import numpy as np
from PIL import Image

from renderer.core import RenderConfig, render as render_bytes
from renderer.fonts import get_approved_font, load_manifest, load_stamp_fonts
from renderer.v3.pipeline import RenderRequest, render
from renderer.v3.stamp import FONT_ORDER, VARIANT_COUNT, deposit_mass, has, resolve

RARE = "\U00020BB7"
EXT_C = "\U0002A700"


def _unstampable_cjk() -> str:
    from renderer.v3.coverage import _CJK_RANGES
    from renderer.v3.stamp import _cmap, _specs

    covered: set[int] = set()
    for spec in _specs():
        covered |= _cmap(str(spec.file))
    for low, high in reversed(_CJK_RANGES):
        for code in range(high, low - 1, -1):
            if code not in covered:
                return chr(code)
    raise AssertionError("every CJK codepoint is stamped")


def test_common_hanzi_is_not_stamped():
    result = render(RenderRequest(text="今天天气很好", max_pages=1, seed=7))
    assert result.metadata["font_stamp_glyphs"] == []
    assert not any("font-stamp" in item for item in result.metadata["fallbacks"])


def test_rare_hanzi_uses_first_handwrite_font():
    expected = resolve(RARE)
    assert expected is not None
    assert expected in FONT_ORDER
    # 巡检：小赖没有，悠哉有，所以不再落到霞鹜印刷楷
    assert expected == "yozai"
    result = render(RenderRequest(text="今天天气很好" + RARE + "我们出去玩", max_pages=1, seed=7))
    assert RARE in result.metadata["font_stamp_glyphs"]
    assert f"{RARE}:font-stamp:{expected}" in result.metadata["fallbacks"]
    assert result.metadata["missing_glyphs"] == []
    page = result.images[0]
    assert page.min() < 200


def test_ext_c_uses_handwrite_then_plangothic():
    font_id = resolve(EXT_C)
    # 巡检：快去写作业有 U+2A700，遍黑也能盖；手写优先所以是 cef-cjk
    assert font_id == "cef-cjk"
    result = render(RenderRequest(text="同学" + EXT_C, max_pages=1, seed=7))
    assert EXT_C in result.metadata["font_stamp_glyphs"]
    assert result.metadata["missing_glyphs"] == []


def test_stamp_mass_has_ink():
    mass, offset = deposit_mass(RARE, em_px=32.0, slant=0.003, aspect=1.12, supersample=2)
    assert mass.shape[0] > 4 and mass.shape[1] > 4
    assert float(mass.max()) > 0.05
    assert isinstance(offset, tuple) and len(offset) == 2


def test_stamp_aspect_changes_mass_ratio():
    square, _ = deposit_mass(RARE, em_px=36.0, aspect=1.0, variant=0, seed=7, supersample=2)
    wide, _ = deposit_mass(RARE, em_px=36.0, aspect=1.12, variant=0, seed=7, supersample=2)
    assert square.shape != wide.shape or not np.array_equal(square, wide)
    square_ratio = square.shape[1] / max(square.shape[0], 1)
    wide_ratio = wide.shape[1] / max(wide.shape[0], 1)
    assert wide_ratio > square_ratio


def test_stamp_variants_differ_but_are_stable():
    a1, _ = deposit_mass(RARE, em_px=32.0, aspect=1.12, variant=0, seed=11, supersample=2)
    a2, _ = deposit_mass(RARE, em_px=32.0, aspect=1.12, variant=0, seed=11, supersample=2)
    b1, _ = deposit_mass(RARE, em_px=32.0, aspect=1.12, variant=1, seed=11, supersample=2)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, b1)
    assert VARIANT_COUNT == 3


def test_unstampable_rare_hanzi_still_fails():
    missing = _unstampable_cjk()
    try:
        render(RenderRequest(text=f"同学{missing}", max_pages=1))
    except ValueError as exc:
        assert missing in str(exc)
        assert "font missing" in str(exc)
    else:
        raise AssertionError("expected missing glyph")
    assert not has(missing)


def test_same_seed_same_bytes_with_stamp():
    text = "今天天气很好" + RARE + "我们出去玩"
    first = render_bytes(text, config=RenderConfig(engine="v3", mode="preview", max_pages=1, seed=11))
    second = render_bytes(text, config=RenderConfig(engine="v3", mode="preview", max_pages=1, seed=11))
    assert first == second
    assert first[:8] == b"\x89PNG\r\n\x1a\n"
    image = Image.open(BytesIO(first))
    arr = np.asarray(image)
    assert arr.min() < 200


def test_style_manifest_hides_stamp_only_fonts():
    ids = [spec.id for spec in load_manifest()]
    assert ids == ["lxgw-wenkai", "yozai", "kose-xiaolai", "mashanzheng"]
    assert get_approved_font("cef-cjk") is None
    assert get_approved_font("plangothic-p1") is None
    stamp_ids = {spec.id for spec in load_stamp_fonts()}
    assert {"cef-cjk", "plangothic-p1", "plangothic-p2", "lxgw-wenkai"} <= stamp_ids
    assert "liujianmaocao" not in stamp_ids
