"""补充骨架：中日韩标点、数字、拉丁字母。

hanzi-writer-data 只覆盖汉字。中文正文离不开"，。"，混排还需要数字与字母，
所以这一套由项目自行编写，坐标约定与骨架库完全一致
（左上原点、y 向下、[0,1] 归一、笔序即书写顺序）。

拉丁字母按"手写印刷体"（manuscript）而非连写草体设计：
中文使用者混排英文时基本都是这个写法。
"""

from __future__ import annotations

import functools
import math

import numpy as np

from .skeleton import GlyphSkeleton
from .strokes import (
    DIAN,
    GOU,
    HENG,
    NA,
    PIE,
    SHU_CHUILU,
    SHU_XUANZHEN,
    TI,
    WAN,
    ZHE,
    profile_for,
)

# 拉丁书写基准线（em 归一坐标，y 向下）
_BASE = 0.80
_XTOP = 0.44
_ASC = 0.14
_DESC = 0.96
_CAP = 0.16


def _arc(
    cx: float, cy: float, rx: float, ry: float, start_deg: float, end_deg: float, count: int = 14
) -> list[tuple[float, float]]:
    angles = np.linspace(math.radians(start_deg), math.radians(end_deg), count)
    return [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in angles]


def _line(x0: float, y0: float, x1: float, y1: float, count: int = 6) -> list[tuple[float, float]]:
    t = np.linspace(0.0, 1.0, count)
    return [(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u) for u in t]


def _dot(cx: float, cy: float, size: float = 0.035) -> list[tuple[float, float]]:
    """一个"点"是短促的下压动作，不是几何圆点。"""
    return _line(cx - size * 0.5, cy - size * 0.6, cx + size * 0.7, cy + size, count=4)


# (笔画点列, 笔型) 的字典。宽度是相对 em 的水平步进。
_GLYPHS: dict[str, tuple[list[tuple[list[tuple[float, float]], int]], float]] = {}


def _define(char: str, strokes: list[tuple[list[tuple[float, float]], int]], advance: float) -> None:
    _GLYPHS[char] = (strokes, advance)


# ------------------------------------------------------------- 中日韩标点

_define("，", [([(0.30, 0.62), (0.39, 0.69), (0.42, 0.77), (0.36, 0.87), (0.25, 0.94)], PIE)], 0.5)
_define("、", [([(0.28, 0.62), (0.35, 0.72), (0.41, 0.85)], DIAN)], 0.5)
_define("。", [(_arc(0.34, 0.78, 0.105, 0.105, -60, 300, 20), WAN)], 0.5)
_define(
    "；",
    [
        (_dot(0.33, 0.46, 0.045), DIAN),
        ([(0.30, 0.66), (0.38, 0.73), (0.40, 0.80), (0.34, 0.90)], PIE),
    ],
    0.5,
)
_define("：", [(_dot(0.33, 0.44, 0.045), DIAN), (_dot(0.33, 0.76, 0.045), DIAN)], 0.5)
_define(
    "？",
    [
        (
            _arc(0.44, 0.32, 0.15, 0.14, 170, 375, 14) + [(0.47, 0.52), (0.45, 0.64)],
            WAN,
        ),
        (_dot(0.44, 0.78, 0.05), DIAN),
    ],
    0.72,
)
_define(
    "！",
    [([(0.44, 0.18), (0.43, 0.42), (0.41, 0.64)], SHU_XUANZHEN), (_dot(0.40, 0.79, 0.05), DIAN)],
    0.55,
)
_define(
    "“",
    [([(0.30, 0.20), (0.26, 0.31)], PIE), ([(0.46, 0.20), (0.42, 0.31)], PIE)],
    0.5,
)
_define(
    "”",
    [([(0.26, 0.20), (0.32, 0.31)], NA), ([(0.42, 0.20), (0.48, 0.31)], NA)],
    0.5,
)
_define("‘", [([(0.36, 0.20), (0.32, 0.31)], PIE)], 0.4)
_define("’", [([(0.32, 0.20), (0.38, 0.31)], NA)], 0.4)
_define("（", [(_arc(0.62, 0.50, 0.30, 0.40, 130, 230, 14), WAN)], 0.5)
_define("）", [(_arc(0.20, 0.50, 0.30, 0.40, 310, 410, 14), WAN)], 0.5)
_define("(", [(_arc(0.60, 0.52, 0.26, 0.34, 130, 230, 14), WAN)], 0.42)
_define(")", [(_arc(0.16, 0.52, 0.26, 0.34, 310, 410, 14), WAN)], 0.42)
_define(
    "《",
    [
        ([(0.62, 0.20), (0.40, 0.46), (0.62, 0.72)], ZHE),
        ([(0.86, 0.24), (0.66, 0.48), (0.86, 0.72)], ZHE),
    ],
    0.5,
)
_define(
    "》",
    [
        ([(0.14, 0.20), (0.36, 0.46), (0.14, 0.72)], ZHE),
        ([(0.38, 0.24), (0.58, 0.48), (0.38, 0.72)], ZHE),
    ],
    0.5,
)
_define("〈", [([(0.60, 0.22), (0.36, 0.48), (0.60, 0.74)], ZHE)], 0.5)
_define("〉", [([(0.20, 0.22), (0.44, 0.48), (0.20, 0.74)], ZHE)], 0.5)
_define("【", [([(0.68, 0.18), (0.30, 0.18), (0.30, 0.80), (0.68, 0.80)], ZHE)], 0.5)
_define("】", [([(0.16, 0.18), (0.54, 0.18), (0.54, 0.80), (0.16, 0.80)], ZHE)], 0.5)
_define("—", [(_line(0.02, 0.50, 0.98, 0.50, 8), HENG)], 1.0)
_define("－", [(_line(0.18, 0.52, 0.82, 0.52, 6), HENG)], 1.0)
_define("…", [(_dot(0.14, 0.72, 0.05), DIAN), (_dot(0.44, 0.72, 0.05), DIAN), (_dot(0.74, 0.72, 0.05), DIAN)], 1.0)
_define("·", [(_dot(0.44, 0.50, 0.05), DIAN)], 0.4)
_define("、", [([(0.28, 0.62), (0.35, 0.72), (0.41, 0.85)], DIAN)], 0.5)
_define(",", [([(0.30, 0.66), (0.38, 0.74), (0.34, 0.90)], PIE)], 0.32)
_define(".", [(_dot(0.32, 0.76, 0.045), DIAN)], 0.30)
_define(";", [(_dot(0.33, 0.52, 0.04), DIAN), ([(0.31, 0.68), (0.38, 0.75), (0.33, 0.90)], PIE)], 0.32)
_define(":", [(_dot(0.33, 0.50, 0.04), DIAN), (_dot(0.33, 0.76, 0.04), DIAN)], 0.30)
_define("!", [([(0.34, 0.18), (0.32, 0.60)], SHU_XUANZHEN), (_dot(0.31, 0.76, 0.045), DIAN)], 0.32)
_define(
    "?",
    [
        (_arc(0.36, 0.30, 0.14, 0.13, 170, 375, 12) + [(0.39, 0.48), (0.37, 0.60)], WAN),
        (_dot(0.36, 0.76, 0.045), DIAN),
    ],
    0.55,
)
_define("-", [(_line(0.10, 0.60, 0.52, 0.59, 5), HENG)], 0.42)
_define("/", [(_line(0.52, 0.16, 0.10, 0.82, 6), PIE)], 0.48)
_define('"', [([(0.22, 0.18), (0.19, 0.32)], PIE), ([(0.40, 0.18), (0.37, 0.32)], PIE)], 0.42)
_define("'", [([(0.24, 0.18), (0.21, 0.32)], PIE)], 0.26)

# ------------------------------------------------------------- 常用符号
# 手写印刷体，放射/轴+箭头，不画五角星或装饰体。

_define(
    "*",
    [
        (_line(0.34, 0.28, 0.34, 0.76, 5), SHU_CHUILU),
        (_line(0.12, 0.52, 0.56, 0.52, 5), HENG),
        (_line(0.18, 0.34, 0.50, 0.70, 5), NA),
        (_line(0.50, 0.34, 0.18, 0.70, 5), PIE),
    ],
    0.58,
)
_define("+", [(_line(0.08, 0.52, 0.56, 0.52, 5), HENG), (_line(0.32, 0.28, 0.32, 0.76, 5), SHU_CHUILU)], 0.58)
_define("=", [(_line(0.10, 0.42, 0.56, 0.42, 5), HENG), (_line(0.10, 0.62, 0.56, 0.62, 5), HENG)], 0.58)
_define(
    "#",
    [
        (_line(0.10, 0.40, 0.58, 0.36, 5), HENG),
        (_line(0.10, 0.64, 0.58, 0.60, 5), HENG),
        (_line(0.24, 0.20, 0.18, 0.82, 5), SHU_CHUILU),
        (_line(0.46, 0.18, 0.40, 0.80, 5), SHU_CHUILU),
    ],
    0.62,
)
_define(
    "@",
    [
        (_arc(0.32, 0.52, 0.22, 0.24, -80, 250, 18), WAN),
        (_arc(0.32, 0.54, 0.10, 0.11, 20, 320, 12) + [(0.44, 0.58), (0.50, 0.70)], WAN),
    ],
    0.64,
)
_define(
    "%",
    [
        (_dot(0.16, 0.28, 0.06), DIAN),
        (_line(0.48, 0.18, 0.12, 0.82, 6), PIE),
        (_dot(0.42, 0.76, 0.06), DIAN),
    ],
    0.58,
)
_define(
    "&",
    [
        (
            _arc(0.28, 0.36, 0.14, 0.14, 200, 420, 12)
            + _arc(0.26, 0.62, 0.18, 0.16, 200, 420, 12)
            + [(0.48, 0.36)],
            WAN,
        ),
        (_line(0.12, 0.72, 0.52, 0.22, 6), TI),
    ],
    0.60,
)
_define("_", [(_line(0.06, 0.84, 0.58, 0.84, 5), HENG)], 0.58)
_define("[", [([(0.42, 0.16), (0.16, 0.16), (0.16, 0.84), (0.42, 0.84)], ZHE)], 0.40)
_define("]", [([(0.10, 0.16), (0.36, 0.16), (0.36, 0.84), (0.10, 0.84)], ZHE)], 0.40)
_define(
    "{",
    [
        (
            [
                (0.42, 0.16),
                (0.26, 0.22),
                (0.22, 0.40),
                (0.10, 0.50),
                (0.22, 0.60),
                (0.26, 0.78),
                (0.42, 0.84),
            ],
            WAN,
        )
    ],
    0.42,
)
_define(
    "}",
    [
        (
            [
                (0.10, 0.16),
                (0.26, 0.22),
                (0.30, 0.40),
                (0.42, 0.50),
                (0.30, 0.60),
                (0.26, 0.78),
                (0.10, 0.84),
            ],
            WAN,
        )
    ],
    0.42,
)
_define("<", [([(0.50, 0.22), (0.14, 0.50), (0.50, 0.78)], ZHE)], 0.48)
_define(">", [([(0.12, 0.22), (0.48, 0.50), (0.12, 0.78)], ZHE)], 0.48)
_define("\\", [(_line(0.12, 0.16, 0.50, 0.82, 6), NA)], 0.48)
_define("|", [(_line(0.28, 0.14, 0.28, 0.86, 6), SHU_CHUILU)], 0.32)
_define(
    "~",
    [
        (
            _arc(0.22, 0.50, 0.12, 0.08, 180, 360, 8)
            + _arc(0.46, 0.50, 0.12, 0.08, 180, 0, 8),
            WAN,
        )
    ],
    0.62,
)
_define("^", [([(0.10, 0.48), (0.30, 0.18), (0.50, 0.48)], ZHE)], 0.52)
_define("`", [(_line(0.18, 0.18, 0.34, 0.34, 4), NA)], 0.30)
_define(
    "$",
    [
        (
            _arc(0.32, 0.36, 0.16, 0.14, 200, 20, 12)
            + _arc(0.32, 0.64, 0.16, 0.14, 20, 200, 12),
            WAN,
        ),
        (_line(0.32, 0.16, 0.32, 0.84, 6), SHU_CHUILU),
    ],
    0.52,
)

# ------------------------------------------------------------------- 箭头
# 第一笔是轴，方向即箭头指向（y 向下）。

_define(
    "→",
    [
        (_line(0.06, 0.52, 0.74, 0.52, 8), HENG),
        (_line(0.74, 0.52, 0.54, 0.34, 5), PIE),
        (_line(0.74, 0.52, 0.54, 0.70, 5), NA),
    ],
    0.84,
)
_define(
    "←",
    [
        (_line(0.74, 0.52, 0.06, 0.52, 8), HENG),
        (_line(0.06, 0.52, 0.26, 0.34, 5), TI),
        (_line(0.06, 0.52, 0.26, 0.70, 5), NA),
    ],
    0.84,
)
_define(
    "↑",
    [
        (_line(0.36, 0.82, 0.36, 0.16, 8), SHU_CHUILU),
        (_line(0.36, 0.16, 0.18, 0.36, 5), PIE),
        (_line(0.36, 0.16, 0.54, 0.36, 5), NA),
    ],
    0.64,
)
_define(
    "↓",
    [
        (_line(0.36, 0.16, 0.36, 0.82, 8), SHU_CHUILU),
        (_line(0.36, 0.82, 0.18, 0.62, 5), PIE),
        (_line(0.36, 0.82, 0.54, 0.62, 5), NA),
    ],
    0.64,
)
_define(
    "↘",
    [
        (_line(0.16, 0.20, 0.70, 0.76, 8), NA),
        (_line(0.70, 0.76, 0.44, 0.74, 4), HENG),
        (_line(0.70, 0.76, 0.68, 0.50, 4), SHU_CHUILU),
    ],
    0.82,
)
_define(
    "↙",
    [
        (_line(0.64, 0.20, 0.12, 0.76, 8), PIE),
        (_line(0.12, 0.76, 0.38, 0.74, 4), HENG),
        (_line(0.12, 0.76, 0.14, 0.50, 4), SHU_CHUILU),
    ],
    0.82,
)
_define(
    "↗",
    [
        (_line(0.16, 0.76, 0.70, 0.20, 8), TI),
        (_line(0.70, 0.20, 0.44, 0.22, 4), HENG),
        (_line(0.70, 0.20, 0.68, 0.46, 4), SHU_CHUILU),
    ],
    0.82,
)
_define(
    "↖",
    [
        (_line(0.64, 0.76, 0.12, 0.20, 8), PIE),
        (_line(0.12, 0.20, 0.38, 0.22, 4), HENG),
        (_line(0.12, 0.20, 0.14, 0.46, 4), SHU_CHUILU),
    ],
    0.82,
)


# ------------------------------------------------------------------- 数字

_define("0", [(_arc(0.30, 0.48, 0.20, 0.32, -90, 270, 18), WAN)], 0.58)
_define("1", [([(0.14, 0.28), (0.28, 0.17)], TI), ([(0.29, 0.17), (0.28, 0.80)], SHU_CHUILU)], 0.5)
_define(
    "2",
    [
        (_arc(0.30, 0.32, 0.19, 0.16, 190, 375, 14) + [(0.44, 0.46), (0.12, 0.79)], WAN),
        (_line(0.10, 0.80, 0.52, 0.79, 5), HENG),
    ],
    0.58,
)
_define(
    "3",
    [
        (
            _arc(0.28, 0.30, 0.18, 0.15, 190, 400, 12)
            + list(_arc(0.26, 0.62, 0.21, 0.19, 285, 460, 14)),
            WAN,
        )
    ],
    0.58,
)
_define("4", [([(0.36, 0.16), (0.08, 0.60), (0.54, 0.60)], ZHE), ([(0.38, 0.36), (0.36, 0.81)], SHU_CHUILU)], 0.58)
_define(
    "5",
    [
        ([(0.48, 0.18), (0.16, 0.18), (0.13, 0.44)], ZHE),
        (_arc(0.28, 0.61, 0.21, 0.20, 200, 430, 16), WAN),
    ],
    0.58,
)
_define(
    "6",
    [([(0.46, 0.18), (0.20, 0.42), (0.12, 0.62)] + list(_arc(0.30, 0.62, 0.19, 0.18, 180, 540, 18)), WAN)],
    0.58,
)
_define("7", [(_line(0.08, 0.19, 0.54, 0.19, 5), HENG), ([(0.54, 0.19), (0.36, 0.50), (0.24, 0.81)], PIE)], 0.58)
_define(
    "8",
    [
        (
            list(_arc(0.30, 0.31, 0.17, 0.15, 90, 450, 16))
            + list(_arc(0.30, 0.62, 0.20, 0.18, 265, 625, 16)),
            WAN,
        )
    ],
    0.58,
)
_define(
    "9",
    [(list(_arc(0.32, 0.34, 0.18, 0.17, 40, 400, 16)) + [(0.48, 0.44), (0.40, 0.66), (0.22, 0.81)], WAN)],
    0.58,
)


# ------------------------------------------------------- 拉丁字母（手写印刷体）

def _latin(char: str, strokes: list[tuple[list[tuple[float, float]], int]], advance: float) -> None:
    _define(char, strokes, advance)


_latin("a", [(_arc(0.26, 0.62, 0.16, 0.18, -20, 300, 16), WAN), ([(0.42, 0.46), (0.41, 0.80)], SHU_CHUILU)], 0.50)
_latin("b", [([(0.10, _ASC), (0.11, _BASE)], SHU_CHUILU), (_arc(0.28, 0.62, 0.17, 0.18, 175, 480, 16), WAN)], 0.50)
_latin("c", [(_arc(0.28, 0.62, 0.17, 0.18, -55, -305, 16), WAN)], 0.46)
_latin("d", [(_arc(0.26, 0.62, 0.16, 0.18, -20, 300, 16), WAN), ([(0.43, _ASC), (0.42, _BASE)], SHU_CHUILU)], 0.50)
_latin("e", [([(0.11, 0.63), (0.44, 0.61)], HENG), (_arc(0.28, 0.62, 0.17, 0.18, 0, 260, 16), WAN)], 0.46)
_latin("f", [(_arc(0.34, 0.28, 0.14, 0.13, 0, 200, 12) + [(0.20, 0.40), (0.18, _BASE)], WAN), (_line(0.06, 0.46, 0.40, 0.45, 4), HENG)], 0.38)
_latin("g", [(_arc(0.26, 0.62, 0.16, 0.18, -20, 300, 16), WAN), ([(0.42, 0.46), (0.41, 0.86), (0.22, _DESC)], WAN)], 0.50)
_latin("h", [([(0.10, _ASC), (0.11, _BASE)], SHU_CHUILU), ([(0.11, 0.58), (0.24, 0.45), (0.40, 0.54), (0.41, _BASE)], WAN)], 0.50)
_latin("i", [([(0.20, 0.46), (0.19, _BASE)], SHU_CHUILU), (_dot(0.20, 0.32, 0.04), DIAN)], 0.28)
_latin("j", [([(0.26, 0.46), (0.24, 0.86), (0.08, _DESC)], WAN), (_dot(0.27, 0.32, 0.04), DIAN)], 0.30)
_latin("k", [([(0.10, _ASC), (0.11, _BASE)], SHU_CHUILU), ([(0.42, 0.46), (0.12, 0.66)], PIE), ([(0.20, 0.60), (0.44, _BASE)], NA)], 0.48)
_latin("l", [([(0.18, _ASC), (0.17, _BASE)], SHU_CHUILU)], 0.26)
_latin("m", [([(0.08, 0.46), (0.09, _BASE)], SHU_CHUILU), ([(0.09, 0.56), (0.20, 0.45), (0.32, 0.54), (0.33, _BASE)], WAN), ([(0.33, 0.56), (0.44, 0.45), (0.56, 0.54), (0.57, _BASE)], WAN)], 0.68)
_latin("n", [([(0.10, 0.46), (0.11, _BASE)], SHU_CHUILU), ([(0.11, 0.56), (0.24, 0.45), (0.40, 0.54), (0.41, _BASE)], WAN)], 0.50)
_latin("o", [(_arc(0.28, 0.62, 0.18, 0.18, -90, 275, 18), WAN)], 0.50)
_latin("p", [([(0.10, 0.46), (0.09, _DESC)], SHU_CHUILU), (_arc(0.28, 0.62, 0.17, 0.18, 175, 480, 16), WAN)], 0.50)
_latin("q", [(_arc(0.26, 0.62, 0.16, 0.18, -20, 300, 16), WAN), ([(0.42, 0.46), (0.41, _DESC)], SHU_CHUILU)], 0.50)
_latin("r", [([(0.12, 0.46), (0.13, _BASE)], SHU_CHUILU), ([(0.13, 0.55), (0.26, 0.45), (0.38, 0.47)], WAN)], 0.36)
_latin("s", [(_arc(0.28, 0.53, 0.14, 0.10, -30, -230, 12) + _arc(0.26, 0.71, 0.15, 0.11, -70, 160, 12), WAN)], 0.42)
_latin("t", [([(0.24, 0.28), (0.22, 0.72), (0.36, _BASE)], WAN), (_line(0.06, 0.46, 0.38, 0.45, 4), HENG)], 0.36)
_latin("u", [([(0.10, 0.46), (0.11, 0.70), (0.24, 0.80), (0.40, 0.68)], WAN), ([(0.41, 0.46), (0.42, _BASE)], SHU_CHUILU)], 0.50)
_latin("v", [([(0.08, 0.46), (0.26, _BASE)], NA), ([(0.26, _BASE), (0.44, 0.46)], TI)], 0.48)
_latin("w", [([(0.06, 0.46), (0.20, _BASE)], NA), ([(0.20, _BASE), (0.32, 0.52)], TI), ([(0.32, 0.52), (0.44, _BASE)], NA), ([(0.44, _BASE), (0.58, 0.46)], TI)], 0.64)
_latin("x", [([(0.08, 0.46), (0.42, _BASE)], NA), ([(0.42, 0.46), (0.08, _BASE)], PIE)], 0.48)
_latin("y", [([(0.08, 0.46), (0.26, 0.78)], NA), ([(0.44, 0.46), (0.16, _DESC)], PIE)], 0.48)
_latin("z", [([(0.08, 0.47), (0.42, 0.47), (0.08, _BASE), (0.44, _BASE)], ZHE)], 0.48)

_latin("A", [([(0.06, _BASE), (0.30, _CAP)], TI), ([(0.30, _CAP), (0.54, _BASE)], NA), (_line(0.14, 0.60, 0.46, 0.60, 4), HENG)], 0.62)
_latin("B", [([(0.10, _CAP), (0.11, _BASE)], SHU_CHUILU), (_arc(0.28, 0.32, 0.16, 0.16, 180, 480, 14), WAN), (_arc(0.28, 0.64, 0.18, 0.17, 180, 480, 14), WAN)], 0.58)
_latin("C", [(_arc(0.32, 0.48, 0.22, 0.32, -50, -310, 18), WAN)], 0.58)
_latin("D", [([(0.10, _CAP), (0.11, _BASE)], SHU_CHUILU), (_arc(0.20, 0.48, 0.30, 0.32, -80, 82, 16), WAN)], 0.60)
_latin("E", [([(0.48, _CAP), (0.12, _CAP), (0.12, _BASE), (0.50, _BASE)], ZHE), (_line(0.13, 0.48, 0.40, 0.48, 4), HENG)], 0.56)
_latin("F", [([(0.48, _CAP), (0.12, _CAP), (0.11, _BASE)], ZHE), (_line(0.13, 0.46, 0.40, 0.46, 4), HENG)], 0.52)
_latin("G", [(_arc(0.32, 0.48, 0.22, 0.32, -50, -320, 18) + [(0.54, 0.52), (0.54, 0.50), (0.36, 0.50)], WAN)], 0.60)
_latin("H", [([(0.10, _CAP), (0.11, _BASE)], SHU_CHUILU), ([(0.48, _CAP), (0.49, _BASE)], SHU_CHUILU), (_line(0.11, 0.50, 0.49, 0.49, 4), HENG)], 0.60)
_latin("I", [([(0.24, _CAP), (0.23, _BASE)], SHU_CHUILU)], 0.34)
_latin("J", [([(0.42, _CAP), (0.40, 0.66)] + list(_arc(0.24, 0.66, 0.16, 0.16, 0, 175, 10)), WAN)], 0.48)
_latin("K", [([(0.10, _CAP), (0.11, _BASE)], SHU_CHUILU), ([(0.50, _CAP), (0.12, 0.52)], PIE), ([(0.24, 0.44), (0.52, _BASE)], NA)], 0.58)
_latin("L", [([(0.12, _CAP), (0.11, _BASE), (0.50, _BASE)], ZHE)], 0.52)
_latin("M", [([(0.06, _BASE), (0.08, _CAP)], SHU_CHUILU), ([(0.08, _CAP), (0.32, 0.56)], NA), ([(0.32, 0.56), (0.56, _CAP)], TI), ([(0.56, _CAP), (0.58, _BASE)], SHU_CHUILU)], 0.68)
_latin("N", [([(0.08, _BASE), (0.10, _CAP)], SHU_CHUILU), ([(0.10, _CAP), (0.48, _BASE)], NA), ([(0.48, _BASE), (0.50, _CAP)], SHU_CHUILU)], 0.60)
_latin("O", [(_arc(0.32, 0.48, 0.24, 0.32, -90, 275, 20), WAN)], 0.64)
_latin("P", [([(0.10, _CAP), (0.11, _BASE)], SHU_CHUILU), (_arc(0.28, 0.34, 0.18, 0.18, 180, 480, 14), WAN)], 0.54)
_latin("Q", [(_arc(0.32, 0.48, 0.24, 0.32, -90, 275, 20), WAN), ([(0.38, 0.66), (0.58, _BASE)], NA)], 0.64)
_latin("R", [([(0.10, _CAP), (0.11, _BASE)], SHU_CHUILU), (_arc(0.28, 0.34, 0.18, 0.18, 180, 480, 14), WAN), ([(0.28, 0.52), (0.54, _BASE)], NA)], 0.58)
_latin("S", [(_arc(0.32, 0.32, 0.18, 0.15, -30, -230, 14) + _arc(0.30, 0.64, 0.20, 0.16, -70, 160, 14), WAN)], 0.54)
_latin("T", [(_line(0.04, _CAP, 0.56, _CAP, 5), HENG), ([(0.31, _CAP), (0.30, _BASE)], SHU_CHUILU)], 0.58)
_latin("U", [([(0.08, _CAP), (0.10, 0.64), (0.30, 0.80), (0.52, 0.62), (0.52, _CAP)], WAN)], 0.60)
_latin("V", [([(0.06, _CAP), (0.30, _BASE)], NA), ([(0.30, _BASE), (0.54, _CAP)], TI)], 0.60)
_latin("W", [([(0.04, _CAP), (0.20, _BASE)], NA), ([(0.20, _BASE), (0.36, 0.36)], TI), ([(0.36, 0.36), (0.52, _BASE)], NA), ([(0.52, _BASE), (0.68, _CAP)], TI)], 0.74)
_latin("X", [([(0.06, _CAP), (0.52, _BASE)], NA), ([(0.52, _CAP), (0.06, _BASE)], PIE)], 0.58)
_latin("Y", [([(0.06, _CAP), (0.30, 0.50)], NA), ([(0.54, _CAP), (0.30, 0.50)], PIE), ([(0.30, 0.50), (0.29, _BASE)], SHU_CHUILU)], 0.60)
_latin("Z", [([(0.06, _CAP), (0.52, _CAP), (0.06, _BASE), (0.54, _BASE)], ZHE)], 0.58)

_ = (GOU, profile_for)


# ------------------------------------------------------- 标点的左右侧空白
#
# 手写时标点不居中：句读紧跟前一个字，空白留在它后面；开引号开括号相反，
# 贴着后一个字，空白留在它前面。之前所有标点的墨迹都落在字身中部，逗号前后
# 各空四分之一字宽，一眼就是排版排出来的 —— 真人不会先空一格再点逗号。
_HUG_PREVIOUS = "，、。；：！？”’）》】]}>」』〕〗"
_HUG_NEXT = "“‘（《【[{<「『〔〖"
_SIDE_BEARING = 0.05


def _shift_x(char: str, offset: float) -> None:
    strokes, advance = _GLYPHS[char]
    _GLYPHS[char] = (
        [([(x + offset, y) for x, y in pts], type_id) for pts, type_id in strokes],
        advance,
    )


def _align_punctuation() -> None:
    for char in _HUG_PREVIOUS + _HUG_NEXT:
        entry = _GLYPHS.get(char)
        if entry is None:
            continue
        xs = [x for pts, _ in entry[0] for x, _ in pts]
        if not xs:
            continue
        if char in _HUG_PREVIOUS:
            _shift_x(char, _SIDE_BEARING - min(xs))
        else:
            _shift_x(char, entry[1] - _SIDE_BEARING - max(xs))


from .extra_symbols import install as _install_extra  # noqa: E402

_install_extra()
_align_punctuation()

from .doodle import install as _install_doodles  # noqa: E402

_install_doodles()


def has(char: str) -> bool:
    return char in _GLYPHS


def advance_of(char: str) -> float:
    entry = _GLYPHS.get(char)
    return entry[1] if entry else 1.0


@functools.lru_cache(maxsize=512)
def get(char: str) -> GlyphSkeleton | None:
    """把补充字形包装成与骨架库一致的 GlyphSkeleton。"""
    entry = _GLYPHS.get(char)
    if entry is None:
        return None
    strokes, _advance = entry
    resampled = []
    type_ids = []
    lengths = []
    weights = []
    for raw, type_id in strokes:
        pts = np.asarray(raw, dtype=np.float32)
        if len(pts) < 2:
            pts = np.repeat(pts, 2, axis=0)
        deltas = np.diff(pts, axis=0)
        seg = np.hypot(deltas[:, 0], deltas[:, 1])
        cumulative = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cumulative[-1])
        targets = np.linspace(0.0, max(total, 1e-6), 24)
        x = np.interp(targets, cumulative, pts[:, 0])
        y = np.interp(targets, cumulative, pts[:, 1])
        resampled.append(np.stack([x, y], axis=1).astype(np.float32))
        type_ids.append(type_id)
        lengths.append(total)
        weights.append(profile_for(type_id).ink_weight)

    points = np.stack(resampled, axis=0)
    merged = points.reshape(-1, 2)
    return GlyphSkeleton(
        char=char,
        points=points,
        type_ids=np.asarray(type_ids, dtype=np.int8),
        has_hook=np.zeros(len(type_ids), dtype=np.int8),
        lengths=np.asarray(lengths, dtype=np.float32),
        ink_weights=np.asarray(weights, dtype=np.float32),
        bbox=(
            float(merged[:, 0].min()),
            float(merged[:, 1].min()),
            float(merged[:, 0].max()),
            float(merged[:, 1].max()),
        ),
        structure={"kind": "single", "groups": [list(range(len(type_ids)))]},
    )


def characters() -> list[str]:
    return sorted(_GLYPHS)
