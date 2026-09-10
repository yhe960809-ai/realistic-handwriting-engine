"""High-frequency characters get more cached writing variants."""

from renderer.v3.paper import CaptureSpec
from renderer.v3.pipeline import (
    PageSpec,
    RenderRequest,
    render,
    variant_budget,
)


def test_variant_budget_keeps_ordinary_chars_at_four():
    budgets = variant_budget({"春": 3, "眠": 3, "的": 2}, ordinary=4)
    assert budgets["春"] == 4
    assert budgets["眠"] == 4
    assert budgets["的"] == 4


def test_variant_budget_boosts_top_frequent_hanzi():
    counts = {char: 100 for char in "的是一人中"}
    counts["永"] = 3
    counts["。"] = 80
    budgets = variant_budget(counts, ordinary=4)
    for char in "的是一人中":
        assert 8 <= budgets[char] <= 12
    assert budgets["永"] == 4
    assert budgets["。"] == 4


def test_repeat_char_stress_reuse_rate():
    text = "的" * 100 + "是" * 100 + "一" * 100 + "人" * 100 + "中" * 100
    result = render(
        RenderRequest(
            text=text,
            page=PageSpec(dpi=72.0, glyph_size_mm=8.0, indent=0),
            capture=CaptureSpec(enabled=False),
            seed=4242,
            supersample=1,
        )
    )
    reuse = result.metadata["variant_reuse"]
    for char in "的是一人中":
        row = reuse[char]
        assert row["count"] >= 8
        assert 8 <= row["budget"] <= 12
        assert 8 <= row["variants"] <= 12
        assert row["variants"] == row["budget"]
        # 100 次、12 变体 → 复用率 0.88；4 变体则是 0.96
        assert row["reuse_rate"] <= 1.0 - 8 / 100
        assert row["reuse_rate"] < 0.96
