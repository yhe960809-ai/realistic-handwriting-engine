"""Renderer 测试。"""

from io import BytesIO

import pytest
from fontTools.ttLib import TTFont
from PIL import Image

from renderer.core import (
    PREVIEW_SCALE,
    RenderConfig,
    _apply_paragraph_indent,
    _fix_punctuation,
    _greedy_wrap,
    _load_font,
    _resolve_font_path,
    _resolve_layout,
    page_size,
    render,
    _wrap,
)


def _char_covered(char: str) -> bool:
    """通过字体 cmap 判断某字符是否有真实 glyph（gid != 0）。"""
    tt = TTFont(str(_resolve_font_path()), fontNumber=0)
    cmap = tt.getBestCmap()
    name = cmap.get(ord(char))
    if name is None:
        return False
    return tt.getGlyphID(name) != 0


def _png_size(data: bytes) -> tuple:
    return Image.open(BytesIO(data)).size


def test_render_chinese_handwriting():
    """验证中文文本可生成手写风格 PNG。"""
    data = render("我能吞下玻璃而不伤身体。")
    assert isinstance(data, bytes)
    assert len(data) > 0
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_supports_multiline():
    data = render("第一行\n第二行\r\n第三行")
    assert isinstance(data, bytes)
    assert len(data) > 0
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_with_config():
    cfg = RenderConfig(
        page_width=800, page_height=600, margin=60, font_size=40,
        line_spacing=60, seed=123,
    )
    data = render("我能吞下玻璃而不伤身体。", config=cfg)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = _png_size(data)
    assert width == 800
    assert height >= 600


def test_seed_determinism():
    """相同 seed 生成像素一致的图片。"""
    text = "我能吞下玻璃而不伤身体。"
    data1 = render(text, config=RenderConfig(seed=42))
    data2 = render(text, config=RenderConfig(seed=42))

    img1 = Image.open(BytesIO(data1))
    img2 = Image.open(BytesIO(data2))
    assert list(img1.tobytes()) == list(img2.tobytes())


def test_different_seed_produces_different_result():
    """不同 seed 生成不同的图片。"""
    text = "我能吞下玻璃而不伤身体。"
    data1 = render(text, config=RenderConfig(seed=1))
    data2 = render(text, config=RenderConfig(seed=2))

    img1 = Image.open(BytesIO(data1))
    img2 = Image.open(BytesIO(data2))
    assert list(img1.tobytes()) != list(img2.tobytes())


def test_long_text_does_not_crash():
    """长文本（约 3000 字）不崩溃、生成 PNG。"""
    text = "我能吞下玻璃而不伤身体。" * 120
    data = render(text)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = _png_size(data)
    assert height > 600


def test_manual_page_break():
    """``---`` 作为手动分页符，生成多页。"""
    text = "第一页内容。\n---\n第二页内容。"
    data = render(text)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    # 多页 PNG 会垂直拼接，高度应大于单页
    width, height = _png_size(data)
    assert height > 600


def test_pdf_output():
    """PDF 格式输出多页。"""
    text = "第一页。\n---\n第二页。"
    cfg = RenderConfig(format="pdf")
    data = render(text, config=cfg)
    assert data[:4] == b"%PDF"
    assert len(data) > 4_000  # 多页 PDF 应明显大于单页


def test_pdf_single_page():
    """PDF 格式输出单页。"""
    cfg = RenderConfig(format="pdf")
    data = render("我能吞下玻璃而不伤身体。", config=cfg)
    assert data[:4] == b"%PDF"
    assert len(data) > 1_000


def test_jpg_output():
    """JPG 格式输出。"""
    cfg = RenderConfig(format="jpg")
    data = render("我能吞下玻璃而不伤身体。", config=cfg)
    assert data[:3] == b"\xff\xd8\xff"  # JPEG magic bytes
    assert len(data) > 1_000


def test_text_too_long_raises():
    """超长文本抛出清晰错误。"""
    try:
        render("字" * 10001)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "too long" in str(e)


def test_page_size_too_large_raises():
    """超大页面尺寸抛出清晰错误。"""
    cfg = RenderConfig(page_width=9999, page_height=9999)
    try:
        render("test", config=cfg)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "too large" in str(e)


def test_max_pages_enforced(monkeypatch):
    """超过最大页数时抛出清晰错误。"""
    import renderer.core as core

    monkeypatch.setattr(core, "MAX_PAGES", 3)
    text = "我能吞下玻璃而不伤身体。" * 200  # 约 2600 字，A4 下远超 3 页
    try:
        render(text)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "too many pages" in str(e)


def test_supported_characters_do_not_raise():
    """常用中文、英文、数字、中文标点均不触发缺字错误。"""
    text = "我能吞下玻璃而不伤身体。Hello, world! 2024 你好，世界。，。、；：？！"
    data = render(text)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_missing_character_raises():
    """字体缺字时返回明确错误，且该字符经 cmap 确认真实缺失。"""
    char = "😀"
    # 自校验：确认该字符在当前字体 cmap 中确实缺失，避免换字体后测试失真
    assert not _char_covered(char), f"{char!r} 应被当前字体视为缺字"
    try:
        render(f"我能吞下玻璃{char}而不伤身体。")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "missing" in str(e).lower()
        assert char in str(e)


def test_preview_is_scaled_down_hd():
    """preview 是 hd 按 PREVIEW_SCALE 等比缩小，布局完全一致。"""
    text = "我能吞下玻璃而不伤身体。这是第二行内容。"
    hd = Image.open(BytesIO(render(text, config=RenderConfig(seed=7, mode="hd"))))
    pv = Image.open(BytesIO(render(text, config=RenderConfig(seed=7, mode="preview"))))

    expected_size = (
        max(1, round(hd.width * PREVIEW_SCALE)),
        max(1, round(hd.height * PREVIEW_SCALE)),
    )
    assert pv.size == expected_size

    # 相同 seed 下，preview 与「hd 等比缩小」像素一致
    expected = hd.resize(pv.size, Image.Resampling.LANCZOS)
    assert list(pv.tobytes()) == list(expected.tobytes())


def test_preview_default_mode_is_hd():
    """默认 mode 为 hd，输出完整尺寸（A4 逻辑尺寸）。"""
    data = render("我能吞下玻璃而不伤身体。", config=RenderConfig(seed=5))
    width, _ = Image.open(BytesIO(data)).size
    assert width == 2100  # 默认 a4 preset page_width


def test_invalid_mode_raises():
    """非法 mode 抛出清晰错误。"""
    cfg = RenderConfig(mode="weird")  # type: ignore[arg-type]
    try:
        render("test", config=cfg)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "mode" in str(e)


def test_right_align_produces_ink_on_right():
    """``>>>`` 右对齐行在页面右侧产生墨迹。"""
    data = render("这是一段正文内容。\n>>>落款：某某")
    img = Image.open(BytesIO(data)).convert("L")
    w, h = img.size

    right_crop = img.crop((int(w * 0.7), 0, w, h))
    assert any(p < 128 for p in right_crop.tobytes()), "右侧区域应有右对齐文字墨迹"


def test_right_align_reaches_right_margin():
    """右对齐行墨迹应贴近右边距（像素级验证）。"""
    data = render("这是一段正文。\n>>>落款：某某")
    img = Image.open(BytesIO(data)).convert("L")
    w, h = img.size

    # 右对齐行的最右墨迹 x 应接近 page_width - margin（1950 附近）
    rightmost = max(x for y in range(h) for x in range(w) if img.getpixel((x, y)) < 128)
    assert rightmost > w * 0.85, f"右对齐最右墨迹 x={rightmost} 应贴近右边距"


# ---------------------------------------------------------------------------
# E1-C 视觉真实性专项
# ---------------------------------------------------------------------------

def test_default_preset_is_a4_ratio():
    """默认纸张为 A4，符合 210:297 比例。"""
    w, h = page_size()
    assert (w, h) == (2100, 2970)
    assert abs(w / h - 210 / 297) < 1e-6


def test_preset_resolves_letter_notebook():
    """letter / notebook 预设可解析为各自尺寸。"""
    from renderer.core import _PAGE_PRESETS

    lw, lh = page_size(RenderConfig(preset="letter"))
    assert (lw, lh) == (
        _PAGE_PRESETS["letter"]["page_width"],
        _PAGE_PRESETS["letter"]["page_height"],
    )
    nw, nh = page_size(RenderConfig(preset="notebook"))
    assert (nw, nh) == (
        _PAGE_PRESETS["notebook"]["page_width"],
        _PAGE_PRESETS["notebook"]["page_height"],
    )


def test_1000_words_fit_in_few_pages():
    """约 1000 字只占 2~4 页（不再是 12 页）。"""
    text = "我能吞下玻璃而不伤身体。" * 77  # 1001 字
    data = render(text, config=RenderConfig(seed=42))
    img = Image.open(BytesIO(data))
    _, page_h = page_size()
    pages = round(img.height / page_h)
    assert 2 <= pages <= 4, f"1000 字应 2~4 页，实际 {pages} 页"


def test_wrap_keeps_english_words_whole():
    """英文单词整词换行，不从中间断开（除非单词本身超宽）。"""
    font = _load_font(75)[0]
    layout = _resolve_layout(RenderConfig())
    max_width = layout.content_width - layout.font_size

    words = ["Hello", "Welcome", "handwriting", "August", "2024"]
    text = "。" + "我".join(words) + "。"
    wrapped = _wrap(text, font, max_width)
    for word in words:
        assert any(word in line for line in wrapped.split("\n")), f"{word} 被拆散"


def test_wrap_splits_overlong_word():
    """单个英文单词本身超过行宽时才允许从中间拆开。"""
    font = _load_font(40)[0]
    max_width = font.getlength("abc")
    lines = _greedy_wrap("abcdefgh", font, max_width)
    assert len(lines) > 1
    for line in lines:
        assert font.getlength(line) <= max_width + 1e-6


def test_punctuation_avoid_line_start():
    """避头：闭标点不得出现在行首。"""
    fixed = _fix_punctuation(["你好世界", "。！"])
    assert all(ln[0] not in "。！" for ln in fixed if ln)


def test_punctuation_avoid_line_end():
    """避尾：开括号不得出现在行尾。"""
    fixed = _fix_punctuation(["你好（", "世界"])
    assert all(ln[-1] not in "（" for ln in fixed if ln)


def test_paragraph_indent_logic():
    """段首缩进：每段首行加全角空格，段内后续行不缩进。"""
    text = "第一段第一行。\n第一段第二行。\n\n第二段。"
    out = _apply_paragraph_indent(text, 2)
    lines = out.split("\n")
    assert lines[0].startswith("　　")
    assert not lines[1].startswith("　")
    assert lines[2] == ""  # 空行保留
    assert lines[3].startswith("　　")  # 第二段首行


def test_paragraph_indent_pixels():
    """段首缩进在像素层面产生约 2 字宽的前导空白。"""
    font = _load_font(75)[0]
    plain = render("这是一个检查段首缩进的段落。", config=RenderConfig(seed=42, paragraph_indent=0))
    indented = render("这是一个检查段首缩进的段落。", config=RenderConfig(seed=42, paragraph_indent=2))

    pimg = Image.open(BytesIO(plain)).convert("L")
    iimg = Image.open(BytesIO(indented)).convert("L")
    w, h = pimg.size

    def leftmost(img):
        return min(x for y in range(0, 300) for x in range(w) if img.getpixel((x, y)) < 128)

    # 缩进版首行墨迹更靠右（缩进了约 2 * font_size）
    assert leftmost(iimg) - leftmost(pimg) >= font.size, "缩进版首行应明显右移"


def test_paper_style_lined_grid():
    """lined / grid 在无墨迹区域产生淡灰格线，plain 为纯白。"""
    plain = Image.open(BytesIO(render("测试", config=RenderConfig(seed=1, paper_style="plain"))))
    lined = Image.open(BytesIO(render("测试", config=RenderConfig(seed=1, paper_style="lined"))))
    grid = Image.open(BytesIO(render("测试", config=RenderConfig(seed=1, paper_style="grid"))))

    # 横线位置 y = margin + line_spacing；x=200 在边距内、缩进前，无墨迹
    y = 150 + 110
    assert plain.getpixel((200, y)) == (255, 255, 255)
    assert lined.getpixel((200, y)) != (255, 255, 255)
    assert grid.getpixel((200, y)) != (255, 255, 255)


def test_ink_variation_reproducible():
    """同 seed + 同 ink_variation 完全复现。"""
    a = render("墨水浓淡变化。", config=RenderConfig(seed=7, ink_variation=0.2))
    b = render("墨水浓淡变化。", config=RenderConfig(seed=7, ink_variation=0.2))
    assert a == b


def test_ink_variation_changes_output():
    """不同 ink_variation 产生不同墨迹（且不改变排版 seed 确定性语义）。"""
    a = render("墨水浓淡变化。", config=RenderConfig(seed=7, ink_variation=0.0))
    b = render("墨水浓淡变化。", config=RenderConfig(seed=7, ink_variation=0.5))
    assert a != b


# ---------------------------------------------------------------------------
# E3-B2 专业参数面板
# ---------------------------------------------------------------------------

_PRO_TEXT = "手写如真专业参数效果测试。相同文字用于比较不同参数对笔迹自然度的影响。"


def _pro_config(**overrides) -> RenderConfig:
    base = dict(seed=20260814, preset="a4", paper_style="plain", format="png", mode="hd")
    base.update(overrides)
    return RenderConfig(**base)


def _preset_neat() -> RenderConfig:
    return _pro_config(
        perturb_x_ratio=1 / 48, perturb_y_ratio=1 / 48, font_size_ratio=1 / 72,
        word_spacing_ratio=1 / 48, line_spacing_ratio=1 / 48,
        perturb_theta_sigma=0.03, ink_variation=0.05,
    )


def _preset_natural() -> RenderConfig:
    return _pro_config(
        perturb_x_ratio=1 / 24, perturb_y_ratio=1 / 24, font_size_ratio=1 / 36,
        word_spacing_ratio=1 / 24, line_spacing_ratio=1 / 24,
        perturb_theta_sigma=0.06, ink_variation=0.1,
    )


def _preset_casual() -> RenderConfig:
    return _pro_config(
        perturb_x_ratio=1 / 16, perturb_y_ratio=1 / 16, font_size_ratio=1 / 24,
        word_spacing_ratio=1 / 16, line_spacing_ratio=1 / 16,
        perturb_theta_sigma=0.09, ink_variation=0.2,
    )


def test_professional_natural_equals_default():
    """「自然」预设（显式 E1-C 常量）与 RenderConfig() 默认输出逐字节一致。"""
    assert render(_PRO_TEXT, _preset_natural()) == render(_PRO_TEXT, _pro_config())


def test_professional_three_presets_differ():
    """工整 / 自然 / 随性三组参数产生互不相同的输出。"""
    neat = render(_PRO_TEXT, _preset_neat())
    natural = render(_PRO_TEXT, _preset_natural())
    casual = render(_PRO_TEXT, _preset_casual())
    assert neat != natural
    assert natural != casual
    assert neat != casual


def test_professional_presets_reproducible():
    """同参数 + 同 seed 逐字节可复现。"""
    for cfg in (_preset_neat(), _preset_natural(), _preset_casual()):
        assert render(_PRO_TEXT, cfg) == render(_PRO_TEXT, cfg)


def test_font_size_scale_changes_output():
    """font_size_scale 改变排版，输出与默认不同。"""
    assert render(_PRO_TEXT, _pro_config()) != render(
        _PRO_TEXT, _pro_config(font_size_scale=1.3)
    )


def test_line_spacing_scale_changes_output():
    """line_spacing_scale 改变排版。"""
    assert render(_PRO_TEXT, _pro_config()) != render(
        _PRO_TEXT, _pro_config(line_spacing_scale=1.4)
    )


def test_word_spacing_changes_output():
    """基础字间距改变排版。"""
    assert render(_PRO_TEXT, _pro_config()) != render(
        _PRO_TEXT, _pro_config(word_spacing=-15)
    )


def test_margin_scale_changes_output():
    """页边距缩放改变排版。"""
    assert render(_PRO_TEXT, _pro_config()) != render(
        _PRO_TEXT, _pro_config(margin_scale=1.5)
    )


@pytest.mark.parametrize(
    "bad",
    [
        dict(font_size_scale=1.5),
        dict(line_spacing_scale=1.7),
        dict(margin_scale=2.0),
        dict(word_spacing=31),
        dict(perturb_x_ratio=0.21),
        dict(perturb_y_ratio=-0.01),
        dict(font_size_ratio=0.3),
        dict(word_spacing_ratio=0.5),
        dict(line_spacing_ratio=1.0),
        dict(perturb_theta_sigma=0.25),
    ],
)
def test_professional_param_bounds_rejected(bad):
    """越界专业参数抛出清晰 ValueError。"""
    try:
        render(_PRO_TEXT, _pro_config(**bad))
        assert False, f"expected ValueError for {bad}"
    except ValueError as e:
        assert "out of range" in str(e)
