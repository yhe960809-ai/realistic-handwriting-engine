"""V3 补充字形：拉丁字母的形状必须可读。

曾经的缺陷：``C`` / ``c`` / ``G`` 的圆弧扫描方向写反，缺口留在字的顶部而不是
右侧，渲染出来 ``C`` 像 ``U``、``G`` 像 ``O``（"AIGC" 被写成 "AIOU"）。
这类错误不会让渲染报错，只会安静地写错字，所以必须用结构断言兜住。
"""

import numpy as np
import pytest

from renderer.v3 import supplement

# em 归一坐标下的字面中心。字形都定义在 x∈[0,0.7]、y∈[0.14,0.96] 附近。
_MID_Y = 0.48


def _stroke_points(char: str) -> np.ndarray:
    skeleton = supplement.get(char)
    assert skeleton is not None, f"{char} 必须有补充字形"
    return skeleton.points.reshape(-1, 2)


def _single_stroke(char: str) -> np.ndarray:
    skeleton = supplement.get(char)
    assert skeleton is not None
    assert skeleton.points.shape[0] == 1, f"{char} 应当是一笔写成"
    return skeleton.points[0]


@pytest.mark.parametrize("char", ["C", "c", "G"])
def test_c_family_opens_to_the_right(char: str) -> None:
    """C 形字母的缺口必须朝右：两个笔画端点都落在字的右半边。"""
    stroke = _single_stroke(char)
    pts = stroke
    center_x = float((pts[:, 0].min() + pts[:, 0].max()) / 2)
    start, end = pts[0], pts[-1]
    assert start[0] > center_x, f"{char} 起笔应在右半边"
    assert end[0] > center_x, f"{char} 收笔应在右半边"


@pytest.mark.parametrize("char", ["C", "c"])
def test_c_family_has_no_ink_on_the_right_edge_midline(char: str) -> None:
    """右侧中线附近必须是空的，否则就闭合成了 O / U。"""
    pts = _stroke_points(char)
    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
    right_edge = x_max - (x_max - x_min) * 0.12
    mid_band = (y_min + y_max) / 2
    band = (y_max - y_min) * 0.12
    on_right_midline = pts[
        (pts[:, 0] >= right_edge) & (np.abs(pts[:, 1] - mid_band) <= band)
    ]
    assert len(on_right_midline) == 0, f"{char} 右侧中线不应有笔迹（缺口开错方向）"


@pytest.mark.parametrize("char", ["O", "o", "0"])
def test_round_letters_stay_closed(char: str) -> None:
    """对照组：O / o / 0 必须闭合，起收笔几乎重合。"""
    stroke = _single_stroke(char)
    span = float(
        max(
            stroke[:, 0].max() - stroke[:, 0].min(),
            stroke[:, 1].max() - stroke[:, 1].min(),
        )
    )
    gap = float(np.hypot(*(stroke[0] - stroke[-1])))
    assert gap < span * 0.35, f"{char} 应当是闭合的圆"


@pytest.mark.parametrize("char", ["S", "s"])
def test_s_reverses_direction_once(char: str) -> None:
    """S 必须是一上一下两段反向弧，只拐一次；断成两团就会糊成星形。"""
    stroke = _single_stroke(char)
    dx = np.diff(stroke[:, 0])
    moving = dx[np.abs(dx) > 1e-4]
    sign_changes = int(np.count_nonzero(np.diff(np.sign(moving)) != 0))
    assert 1 <= sign_changes <= 3, f"{char} 水平方向应恰好折返一次，实测 {sign_changes}"


def test_every_ascii_letter_and_digit_is_defined() -> None:
    """缺字会被安静地跳过（不报错），所以覆盖率要显式断言。"""
    required = (
        [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        + [chr(c) for c in range(ord("a"), ord("z") + 1)]
        + [chr(c) for c in range(ord("0"), ord("9") + 1)]
    )
    missing = [c for c in required if not supplement.has(c)]
    assert missing == [], f"缺少补充字形: {missing}"


@pytest.mark.parametrize(
    "char, expected_strokes",
    [("A", 3), ("B", 3), ("C", 1), ("H", 3), ("O", 1), ("T", 2), ("i", 2)],
)
def test_stroke_counts_match_handwriting(char: str, expected_strokes: int) -> None:
    """笔画数写错通常意味着字形被改坏了。"""
    skeleton = supplement.get(char)
    assert skeleton is not None
    assert skeleton.points.shape[0] == expected_strokes
