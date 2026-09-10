"""V3 常用符号与箭头：覆盖、方向、用户原文、缺字仍响。"""

from io import BytesIO

import pytest
from PIL import Image

from renderer.core import RenderConfig, render
from renderer.v3 import supplement
from renderer.v3.coverage import normalize_text
from renderer.v3.pipeline import RenderRequest
from renderer.v3.pipeline import render as render_v3

REQUIRED_SYMBOLS = (
    list("*+=#@%&_[]{}<>\\|~^`$")
    + list("←→↑↓↖↗↘↙")
    + list("♥♡❤★☆✓✔✗")
    + list("￥¥℃°×÷±≈≠√「」『』～％※○●□■△")
    + list("①②③④⑤⑥⑦⑧⑨⑩")
    + list("⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳")
    + list("⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽")
    + list("㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩")
    + list("ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ")
    + list("≤≥∞∵∴∠⊥≡")
    + list("〔〕〖〗")
    + list("▽▲▼◇◆")
    + list("℉‰㎡")
    + list("⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇")
    + list("⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑")
)


def _shaft(char: str):
    skeleton = supplement.get(char)
    assert skeleton is not None, f"{char} 必须有补充字形"
    return skeleton.points[0]


@pytest.mark.parametrize("char", REQUIRED_SYMBOLS)
def test_each_symbol_is_defined(char: str) -> None:
    assert supplement.has(char), f"缺少补充字形: {char!r}"
    assert supplement.get(char) is not None


def test_right_arrow_shaft_ends_to_the_right() -> None:
    stroke = _shaft("→")
    assert float(stroke[-1, 0]) > float(stroke[0, 0])


def test_southeast_arrow_shaft_ends_right_and_down() -> None:
    stroke = _shaft("↘")
    assert float(stroke[-1, 0]) > float(stroke[0, 0])
    assert float(stroke[-1, 1]) > float(stroke[0, 1])


def test_user_text_with_star_and_arrows_renders() -> None:
    data = render(
        "步骤1*注意→右下↘",
        config=RenderConfig(engine="v3", mode="preview", max_pages=1),
    )
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    image = Image.open(BytesIO(data))
    assert image.size[0] > 0 and image.size[1] > 0


def test_emoji_still_raises_missing() -> None:
    with pytest.raises(ValueError, match="missing") as exc:
        render("你好😀", config=RenderConfig(engine="v3", mode="preview", max_pages=1))
    assert "😀" in str(exc.value)


def test_phone_heart_and_star_render() -> None:
    data = render(
        "加油❤⭐✓",
        config=RenderConfig(engine="v3", mode="preview", max_pages=1),
    )
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    image = Image.open(BytesIO(data))
    assert image.size[0] > 0


def test_variation_selector_heart_renders() -> None:
    data = render(
        "喜欢❤️谢谢",
        config=RenderConfig(engine="v3", mode="preview", max_pages=1),
    )
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_homework_symbols_mix_renders_without_stamp() -> None:
    text = "应付￥12×3≈36℃ 「备注」①"
    data = render(text, config=RenderConfig(engine="v3", mode="preview", max_pages=1))
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    image = Image.open(BytesIO(data))
    assert image.size[0] > 0 and image.size[1] > 0
    result = render_v3(RenderRequest(text=text, max_pages=1, seed=7))
    assert result.metadata["font_stamp_glyphs"] == []
    assert not any("font-stamp" in item for item in result.metadata["fallbacks"])


def test_wave_dash_normalizes_and_renders() -> None:
    assert normalize_text("区间〜两端") == "区间～两端"
    data = render(
        "区间〜两端",
        config=RenderConfig(engine="v3", mode="preview", max_pages=1),
    )
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    image = Image.open(BytesIO(data))
    assert image.size[0] > 0


def test_homework_aliases_normalize() -> None:
    assert normalize_text("＋０＝３") == "+0=3"
    assert normalize_text("1–2⋯结束") == "1-2…结束"
    assert normalize_text("ⅰ✕☑") == "Ⅰ×✓"


def test_homework_serials_and_math_render() -> None:
    text = "⑴②Ⅰ 〔注〕x≤3∞ ㎡ ⑪"
    data = render(text, config=RenderConfig(engine="v3", mode="preview", max_pages=1))
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    result = render_v3(RenderRequest(text=text, max_pages=1, seed=7))
    assert result.metadata["font_stamp_glyphs"] == []
    assert not any("font-stamp" in item for item in result.metadata["fallbacks"])
