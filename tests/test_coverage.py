"""覆盖扫描：分类、归一、缺字预检。不渲整页（最后一项只走 fail-fast）。"""

import os

from renderer.core import _missing_message
from renderer.v3.coverage import (
    classify_cluster,
    normalize_text,
    scan_text,
    unsupported_clusters,
)
from renderer.v3.pipeline import RenderRequest, render
from renderer.v3.stamp import has as stamp_has

UNSTAMPABLE = "\uFAFF"
RARE = "\U00020BB7"
EXT_C = "\U0002A700"


def test_normalize_is_identity_for_plain_chinese():
    text = "春天的早晨，阳光洒在窗台上。"
    assert normalize_text(text) == text
    assert unsupported_clusters(text) == []


def test_heart_sequence_normalizes_and_is_supported():
    assert normalize_text("喜欢\u2764\ufe0f") == "喜欢\u2764"
    report = scan_text("喜欢\u2764\ufe0f")
    assert report.unsupported == ()
    assert any(hit.cluster == "\u2764" and hit.category == "doodle" for hit in report.supported)


def test_star_alias_is_supported():
    assert normalize_text("\u2b50") == "\u2605"
    assert unsupported_clusters("看\u2b50") == []


def test_fullwidth_and_math_aliases_are_supported():
    assert normalize_text("＋１２") == "+12"
    assert unsupported_clusters("＋１２≤３") == []
    assert unsupported_clusters("ⅰ–ⅹ") == []


def test_rare_hanzi_ext_b_is_rare_and_stampable():
    assert classify_cluster(RARE) == "rare_hanzi"
    assert stamp_has(RARE)
    report = scan_text("同学" + RARE)
    assert RARE not in [hit.cluster for hit in report.unsupported]
    assert any(hit.cluster == RARE and hit.category == "rare_hanzi" for hit in report.stamped)


def test_emoji_is_emoji_category():
    report = scan_text("你好\U0001F600")
    assert len(report.unsupported) == 1
    assert report.unsupported[0].cluster == "\U0001F600"
    assert report.unsupported[0].category == "emoji"
    assert report.unsupported[0].codepoint == "U+1F600"


def test_fail_fast_message_matches_core_contract():
    missing = unsupported_clusters("有\U0001F600")
    assert _missing_message(missing) == "font missing 1 character(s): \U0001F600"


def test_ext_c_is_stamped_not_missing():
    assert classify_cluster(EXT_C) == "rare_hanzi"
    assert stamp_has(EXT_C)
    report = scan_text("同学" + EXT_C)
    assert EXT_C not in [hit.cluster for hit in report.unsupported]
    assert any(hit.cluster == EXT_C for hit in report.stamped)


def test_compat_ideograph_aliases_when_stamp_off():
    os.environ["HANDWRITE_V3_FONT_STAMP"] = "0"
    try:
        assert normalize_text("\uf90a") == "金"
        assert unsupported_clusters("\uf90a") == []
    finally:
        os.environ.pop("HANDWRITE_V3_FONT_STAMP", None)


def test_new_homework_numerals_are_supported():
    assert unsupported_clusters("⑾⒈") == []


def test_fail_fast_skips_full_render_when_font_also_missing():
    try:
        render(RenderRequest(text=f"签名：{UNSTAMPABLE}", max_pages=1))
    except ValueError as exc:
        assert UNSTAMPABLE in str(exc)
        assert "font missing" in str(exc)
    else:
        raise AssertionError("expected missing glyph")


def test_flag_off_restores_fail_fast_for_stampable_rare_hanzi():
    os.environ["HANDWRITE_V3_FONT_STAMP"] = "0"
    try:
        assert RARE in unsupported_clusters("同学" + RARE)
        try:
            render(RenderRequest(text="签名：" + RARE, max_pages=1))
        except ValueError as exc:
            assert RARE in str(exc)
        else:
            raise AssertionError("expected missing glyph")
    finally:
        os.environ.pop("HANDWRITE_V3_FONT_STAMP", None)
