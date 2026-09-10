"""v3 文本标记与拉丁空格：不走整页渲染，只测排版预处理。"""

from renderer.v3.pipeline import (
    _PAGE_BREAK,
    _advance_em,
    _page_count,
    _wrap,
)


def test_ascii_space_is_narrower_than_hanzi():
    assert _advance_em(" ") == 0.32
    assert _advance_em("中") == 1.0
    assert _advance_em("a") < 1.0


def test_wrap_right_align_and_manual_page_break():
    lines = _wrap("今天交 essay\n>>>落款：某某\n---\n第二页", columns_em=20, indent=0)
    chars, aligns = zip(*lines)
    assert "right" in aligns
    right_line = next(cs for cs, al in lines if al == "right")
    assert "落" in right_line
    assert ">>>" not in "".join(right_line)
    assert any(cs == [_PAGE_BREAK] for cs in chars)
    assert _page_count(lines, 10) >= 2
