"""作业常用标点与符号。坐标约定与 supplement 相同。

禁止覆盖已有字形。由 supplement 在贴边之前调用一次 install()。
"""

from __future__ import annotations

from .strokes import DIAN, HENG, NA, PIE, SHU_CHUILU, WAN, ZHE
from .supplement import _GLYPHS, _arc, _define, _dot, _line

_NEW = (
    "￥¥℃°×÷±≈≠√「」『』～％※○●□■△"
    "①②③④⑤⑥⑦⑧⑨⑩"
    "⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
    "⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽"
    "㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩"
    "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ"
    "≤≥∞∵∴∠⊥≡"
    "〔〕〖〗"
    "▽▲▼◇◆"
    "℉‰㎡"
    "⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇"
    "⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑"
)


def _add(char: str, strokes: list, advance: float) -> None:
    if char in _GLYPHS:
        raise ValueError(f"refusing to overwrite existing glyph {char!r}")
    _define(char, strokes, advance)


def _box(points: list[tuple[float, float]], x0: float, y0: float, x1: float, y1: float):
    return [(x0 + (x1 - x0) * x, y0 + (y1 - y0) * y) for x, y in points]


def _digit_strokes(digit: str) -> list[tuple[list[tuple[float, float]], int]]:
    """单位正方形里的手写数字，供圈码缩小放入。"""
    if digit == "0":
        return [(_box(list(_arc(0.50, 0.50, 0.28, 0.36, -90, 270, 14)), 0, 0, 1, 1), WAN)]
    if digit == "1":
        return [(_box([(0.35, 0.22), (0.52, 0.12), (0.50, 0.88)], 0, 0, 1, 1), SHU_CHUILU)]
    if digit == "2":
        return [
            (
                _box(
                    [(0.22, 0.28), (0.38, 0.12), (0.72, 0.18), (0.70, 0.38), (0.28, 0.88), (0.78, 0.88)],
                    0,
                    0,
                    1,
                    1,
                ),
                ZHE,
            )
        ]
    if digit == "3":
        return [
            (
                _box(
                    [
                        (0.24, 0.18),
                        (0.70, 0.16),
                        (0.62, 0.46),
                        (0.36, 0.50),
                        (0.68, 0.58),
                        (0.62, 0.86),
                        (0.24, 0.84),
                    ],
                    0,
                    0,
                    1,
                    1,
                ),
                WAN,
            )
        ]
    if digit == "4":
        return [
            (_box([(0.58, 0.12), (0.22, 0.62), (0.78, 0.62)], 0, 0, 1, 1), ZHE),
            (_box([(0.60, 0.12), (0.58, 0.88)], 0, 0, 1, 1), SHU_CHUILU),
        ]
    if digit == "5":
        return [
            (_box([(0.70, 0.14), (0.28, 0.14), (0.26, 0.46), (0.68, 0.50)], 0, 0, 1, 1), ZHE),
            (_box([(0.68, 0.50), (0.70, 0.82), (0.28, 0.86)], 0, 0, 1, 1), WAN),
        ]
    if digit == "6":
        return [
            (
                _box(
                    [
                        (0.64, 0.16),
                        (0.30, 0.22),
                        (0.24, 0.52),
                        (0.28, 0.82),
                        (0.62, 0.86),
                        (0.70, 0.60),
                        (0.36, 0.54),
                    ],
                    0,
                    0,
                    1,
                    1,
                ),
                WAN,
            )
        ]
    if digit == "7":
        return [(_box([(0.22, 0.16), (0.78, 0.16), (0.42, 0.88)], 0, 0, 1, 1), ZHE)]
    if digit == "8":
        return [
            (_box(list(_arc(0.50, 0.32, 0.22, 0.16, -90, 270, 12)), 0, 0, 1, 1), WAN),
            (_box(list(_arc(0.50, 0.68, 0.24, 0.18, -90, 270, 12)), 0, 0, 1, 1), WAN),
        ]
    if digit == "9":
        return [
            (
                _box(
                    [
                        (0.68, 0.48),
                        (0.66, 0.20),
                        (0.32, 0.16),
                        (0.26, 0.40),
                        (0.58, 0.48),
                        (0.70, 0.22),
                    ],
                    0,
                    0,
                    1,
                    1,
                ),
                WAN,
            ),
            (_box([(0.66, 0.40), (0.58, 0.88)], 0, 0, 1, 1), PIE),
        ]
    if digit == "10":
        return [
            (_box([(0.18, 0.20), (0.28, 0.14), (0.26, 0.86)], 0, 0, 1, 1), SHU_CHUILU),
            (_box(list(_arc(0.62, 0.50, 0.20, 0.30, -90, 270, 14)), 0, 0, 1, 1), WAN),
        ]
    raise KeyError(digit)


def _number_strokes(value: str) -> list[tuple[list[tuple[float, float]], int]]:
    if value == "10" or (len(value) == 1 and value in "0123456789"):
        return _digit_strokes(value)
    if value == "20":
        tens, ones = "2", "0"
    elif len(value) == 2 and value[0] == "1":
        tens, ones = "1", value[1]
    else:
        raise KeyError(value)
    packed: list[tuple[list[tuple[float, float]], int]] = []
    for pts, type_id in _digit_strokes(tens):
        packed.append((_box(pts, 0.00, 0.05, 0.46, 0.95), type_id))
    for pts, type_id in _digit_strokes(ones):
        packed.append((_box(pts, 0.50, 0.05, 0.96, 0.95), type_id))
    return packed


def _circled(value: str) -> list[tuple[list[tuple[float, float]], int]]:
    ring = (_arc(0.50, 0.50, 0.40, 0.40, -90, 270, 22), WAN)
    x0, x1 = (0.30, 0.70) if len(value) == 1 or value == "10" else (0.24, 0.76)
    inner = [(_box(pts, x0, 0.26, x1, 0.74), type_id) for pts, type_id in _number_strokes(value)]
    return [ring, *inner]


def _paren_num(value: str) -> list[tuple[list[tuple[float, float]], int]]:
    left = (_arc(0.16, 0.50, 0.12, 0.38, 130, 230, 12), WAN)
    right = (_arc(0.84, 0.50, 0.12, 0.38, 310, 410, 12), WAN)
    x0, x1 = (0.28, 0.72) if len(value) == 1 or value == "10" else (0.24, 0.76)
    inner = [(_box(pts, x0, 0.22, x1, 0.78), type_id) for pts, type_id in _number_strokes(value)]
    return [left, *inner, right]


def _dotted_num(value: str) -> list[tuple[list[tuple[float, float]], int]]:
    if value == "10":
        inner = [(_box(pts, 0.04, 0.16, 0.70, 0.84), type_id) for pts, type_id in _number_strokes(value)]
    else:
        inner = [(_box(pts, 0.10, 0.16, 0.64, 0.84), type_id) for pts, type_id in _number_strokes(value)]
    return [*inner, (_dot(0.82, 0.78, 0.05), DIAN)]


def _cn_strokes(value: str) -> list[tuple[list[tuple[float, float]], int]]:
    if value == "1":
        return [(_line(0.18, 0.50, 0.82, 0.50, 5), HENG)]
    if value == "2":
        return [
            (_line(0.20, 0.36, 0.80, 0.36, 5), HENG),
            (_line(0.18, 0.66, 0.82, 0.66, 5), HENG),
        ]
    if value == "3":
        return [
            (_line(0.22, 0.28, 0.78, 0.28, 5), HENG),
            (_line(0.20, 0.50, 0.80, 0.50, 5), HENG),
            (_line(0.18, 0.72, 0.82, 0.72, 5), HENG),
        ]
    if value == "4":
        return [
            ([(0.22, 0.22), (0.78, 0.22), (0.78, 0.80), (0.22, 0.80), (0.22, 0.22)], ZHE),
            (_line(0.40, 0.22, 0.36, 0.80, 5), SHU_CHUILU),
            (_line(0.62, 0.38, 0.62, 0.80, 5), SHU_CHUILU),
        ]
    if value == "5":
        return [
            (_line(0.22, 0.22, 0.78, 0.22, 5), HENG),
            ([(0.38, 0.22), (0.32, 0.50), (0.72, 0.52)], ZHE),
            (_line(0.20, 0.78, 0.80, 0.78, 5), HENG),
        ]
    if value == "6":
        return [
            (_dot(0.50, 0.20, 0.04), DIAN),
            (_line(0.22, 0.36, 0.78, 0.36, 5), HENG),
            (_line(0.28, 0.36, 0.18, 0.82, 5), PIE),
            (_line(0.72, 0.36, 0.82, 0.82, 5), NA),
        ]
    if value == "7":
        return [
            (_line(0.20, 0.28, 0.80, 0.28, 5), HENG),
            ([(0.48, 0.28), (0.42, 0.80)], SHU_CHUILU),
        ]
    if value == "8":
        return [
            (_line(0.22, 0.22, 0.78, 0.82, 6), NA),
            (_line(0.78, 0.22, 0.22, 0.82, 6), PIE),
        ]
    if value == "9":
        return [
            (_line(0.28, 0.22, 0.22, 0.70, 5), PIE),
            ([(0.28, 0.22), (0.70, 0.22), (0.66, 0.50), (0.36, 0.52)], ZHE),
            (_arc(0.48, 0.62, 0.22, 0.20, -20, 200, 10), WAN),
        ]
    if value == "10":
        return [
            (_line(0.18, 0.50, 0.82, 0.50, 5), HENG),
            (_line(0.50, 0.18, 0.50, 0.84, 5), SHU_CHUILU),
        ]
    raise KeyError(value)


def _paren_cn(value: str) -> list[tuple[list[tuple[float, float]], int]]:
    left = (_arc(0.16, 0.50, 0.12, 0.38, 130, 230, 12), WAN)
    right = (_arc(0.84, 0.50, 0.12, 0.38, 310, 410, 12), WAN)
    inner = [(_box(pts, 0.28, 0.22, 0.72, 0.78), type_id) for pts, type_id in _cn_strokes(value)]
    return [left, *inner, right]


def _place_unit(
    strokes: list[tuple[list[tuple[float, float]], int]], x0: float, x1: float
) -> list[tuple[list[tuple[float, float]], int]]:
    return [(_box(pts, x0, 0.0, x1, 1.0), type_id) for pts, type_id in strokes]


def _roman_strokes(n: int) -> list[tuple[list[tuple[float, float]], int]]:
    i = [(_line(0.50, 0.16, 0.50, 0.84, 6), SHU_CHUILU)]
    v = [
        (_line(0.18, 0.16, 0.50, 0.84, 6), NA),
        (_line(0.50, 0.84, 0.82, 0.16, 6), PIE),
    ]
    x = [
        (_line(0.18, 0.16, 0.82, 0.84, 6), NA),
        (_line(0.82, 0.16, 0.18, 0.84, 6), PIE),
    ]
    if n == 1:
        return i
    if n == 2:
        return _place_unit(i, 0.28, 0.48) + _place_unit(i, 0.52, 0.72)
    if n == 3:
        return _place_unit(i, 0.18, 0.38) + _place_unit(i, 0.40, 0.60) + _place_unit(i, 0.62, 0.82)
    if n == 4:
        return _place_unit(i, 0.06, 0.34) + _place_unit(v, 0.34, 0.98)
    if n == 5:
        return v
    if n == 6:
        return _place_unit(v, 0.02, 0.60) + _place_unit(i, 0.66, 0.90)
    if n == 7:
        return _place_unit(v, 0.00, 0.50) + _place_unit(i, 0.54, 0.72) + _place_unit(i, 0.76, 0.94)
    if n == 8:
        return (
            _place_unit(v, 0.00, 0.42)
            + _place_unit(i, 0.44, 0.60)
            + _place_unit(i, 0.62, 0.78)
            + _place_unit(i, 0.80, 0.96)
        )
    if n == 9:
        return _place_unit(i, 0.04, 0.32) + _place_unit(x, 0.30, 0.98)
    if n == 10:
        return x
    raise KeyError(n)


def install() -> None:
    _add(
        "￥",
        [
            ([(0.18, 0.22), (0.50, 0.48)], NA),
            ([(0.82, 0.22), (0.50, 0.48)], PIE),
            ([(0.50, 0.48), (0.50, 0.86)], SHU_CHUILU),
            (_line(0.28, 0.56, 0.72, 0.56, 5), HENG),
            (_line(0.28, 0.68, 0.72, 0.68, 5), HENG),
        ],
        0.92,
    )
    _add(
        "¥",
        [
            ([(0.16, 0.20), (0.50, 0.50)], NA),
            ([(0.84, 0.20), (0.50, 0.50)], PIE),
            ([(0.50, 0.50), (0.50, 0.86)], SHU_CHUILU),
            (_line(0.30, 0.58, 0.70, 0.58, 5), HENG),
            (_line(0.30, 0.70, 0.70, 0.70, 5), HENG),
        ],
        0.86,
    )
    _add(
        "℃",
        [
            (_arc(0.22, 0.28, 0.10, 0.10, -90, 270, 12), WAN),
            (_arc(0.58, 0.54, 0.24, 0.28, -50, -310, 16), WAN),
        ],
        0.92,
    )
    _add("°", [(_arc(0.36, 0.28, 0.12, 0.12, -90, 270, 12), WAN)], 0.42)
    _add(
        "×",
        [
            (_line(0.22, 0.26, 0.78, 0.78, 6), NA),
            (_line(0.78, 0.26, 0.22, 0.78, 6), PIE),
        ],
        0.78,
    )
    _add(
        "÷",
        [
            (_dot(0.50, 0.30, 0.045), DIAN),
            (_line(0.22, 0.52, 0.78, 0.52, 6), HENG),
            (_dot(0.50, 0.74, 0.045), DIAN),
        ],
        0.78,
    )
    _add(
        "±",
        [
            (_line(0.22, 0.40, 0.78, 0.40, 5), HENG),
            (_line(0.50, 0.18, 0.50, 0.62, 5), SHU_CHUILU),
            (_line(0.22, 0.78, 0.78, 0.78, 5), HENG),
        ],
        0.78,
    )
    _add(
        "≈",
        [
            (
                _arc(0.34, 0.40, 0.16, 0.08, 180, 360, 8)
                + _arc(0.66, 0.40, 0.16, 0.08, 180, 0, 8),
                WAN,
            ),
            (
                _arc(0.34, 0.62, 0.16, 0.08, 180, 360, 8)
                + _arc(0.66, 0.62, 0.16, 0.08, 180, 0, 8),
                WAN,
            ),
        ],
        0.82,
    )
    _add(
        "≠",
        [
            (_line(0.20, 0.38, 0.80, 0.38, 5), HENG),
            (_line(0.20, 0.62, 0.80, 0.62, 5), HENG),
            (_line(0.70, 0.18, 0.30, 0.82, 6), PIE),
        ],
        0.80,
    )
    _add(
        "√",
        [
            ([(0.12, 0.52), (0.30, 0.78), (0.86, 0.18)], NA),
            (_line(0.86, 0.18, 0.86, 0.30, 4), SHU_CHUILU),
        ],
        0.86,
    )
    _add(
        "「",
        [
            ([(0.72, 0.18), (0.28, 0.18), (0.28, 0.62)], ZHE),
        ],
        0.50,
    )
    _add(
        "」",
        [
            ([(0.28, 0.82), (0.72, 0.82), (0.72, 0.38)], ZHE),
        ],
        0.50,
    )
    _add(
        "『",
        [
            ([(0.78, 0.16), (0.24, 0.16), (0.24, 0.66)], ZHE),
            ([(0.66, 0.28), (0.38, 0.28), (0.38, 0.58)], ZHE),
        ],
        0.55,
    )
    _add(
        "』",
        [
            ([(0.22, 0.84), (0.76, 0.84), (0.76, 0.34)], ZHE),
            ([(0.34, 0.72), (0.62, 0.72), (0.62, 0.42)], ZHE),
        ],
        0.55,
    )
    _add(
        "～",
        [
            (
                _arc(0.30, 0.50, 0.16, 0.12, 180, 360, 8)
                + _arc(0.62, 0.50, 0.16, 0.12, 180, 0, 8),
                WAN,
            )
        ],
        0.90,
    )
    _add(
        "％",
        [
            (_dot(0.28, 0.28, 0.06), DIAN),
            (_line(0.72, 0.20, 0.22, 0.82, 6), PIE),
            (_dot(0.70, 0.74, 0.06), DIAN),
        ],
        0.88,
    )
    _add(
        "※",
        [
            (_line(0.22, 0.22, 0.78, 0.78, 6), NA),
            (_line(0.78, 0.22, 0.22, 0.78, 6), PIE),
            (_line(0.50, 0.16, 0.50, 0.84, 6), SHU_CHUILU),
            (_line(0.16, 0.50, 0.84, 0.50, 6), HENG),
        ],
        0.90,
    )
    _add("○", [(_arc(0.50, 0.50, 0.32, 0.32, -90, 270, 20), WAN)], 0.86)
    _add(
        "●",
        [
            (_arc(0.50, 0.50, 0.30, 0.30, -90, 270, 18), WAN),
            (_arc(0.50, 0.50, 0.16, 0.16, -90, 270, 12), WAN),
            (_line(0.32, 0.32, 0.68, 0.68, 5), NA),
            (_line(0.68, 0.32, 0.32, 0.68, 5), PIE),
        ],
        0.86,
    )
    _add(
        "□",
        [
            (
                [(0.22, 0.22), (0.78, 0.22), (0.78, 0.78), (0.22, 0.78), (0.22, 0.22)],
                ZHE,
            )
        ],
        0.86,
    )
    _add(
        "■",
        [
            ([(0.22, 0.22), (0.78, 0.22), (0.78, 0.78), (0.22, 0.78), (0.22, 0.22)], ZHE),
            (_line(0.32, 0.32, 0.68, 0.68, 5), NA),
            (_line(0.68, 0.32, 0.32, 0.68, 5), PIE),
        ],
        0.86,
    )
    _add(
        "△",
        [
            ([(0.50, 0.16), (0.18, 0.82), (0.82, 0.82), (0.50, 0.16)], ZHE),
        ],
        0.86,
    )

    circled = "①②③④⑤⑥⑦⑧⑨⑩"
    digits = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")
    for char, digit in zip(circled, digits, strict=True):
        _add(char, _circled(digit), 0.92)
    for char, value in zip("⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳", (str(n) for n in range(11, 21)), strict=True):
        _add(char, _circled(value), 0.92)
    for char, value in zip("⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽", digits, strict=True):
        _add(char, _paren_num(value), 0.92)
    for char, value in zip("⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇", (str(n) for n in range(11, 21)), strict=True):
        _add(char, _paren_num(value), 0.92)
    for char, value in zip("⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑", digits, strict=True):
        _add(char, _dotted_num(value), 0.86)
    for char, value in zip("㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩", digits, strict=True):
        _add(char, _paren_cn(value), 0.92)
    for char, n in zip("ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ", range(1, 11), strict=True):
        _add(char, _roman_strokes(n), 0.86)

    _add(
        "≤",
        [
            ([(0.72, 0.22), (0.22, 0.40), (0.72, 0.58)], ZHE),
            (_line(0.22, 0.78, 0.78, 0.78, 5), HENG),
        ],
        0.80,
    )
    _add(
        "≥",
        [
            ([(0.22, 0.22), (0.72, 0.40), (0.22, 0.58)], ZHE),
            (_line(0.22, 0.78, 0.78, 0.78, 5), HENG),
        ],
        0.80,
    )
    _add(
        "∞",
        [
            (
                _arc(0.32, 0.50, 0.18, 0.16, 0, 360, 14)
                + _arc(0.68, 0.50, 0.18, 0.16, 180, 540, 14),
                WAN,
            )
        ],
        0.96,
    )
    _add(
        "∵",
        [
            (_dot(0.50, 0.28, 0.05), DIAN),
            (_dot(0.28, 0.72, 0.05), DIAN),
            (_dot(0.72, 0.72, 0.05), DIAN),
        ],
        0.78,
    )
    _add(
        "∴",
        [
            (_dot(0.28, 0.28, 0.05), DIAN),
            (_dot(0.72, 0.28, 0.05), DIAN),
            (_dot(0.50, 0.72, 0.05), DIAN),
        ],
        0.78,
    )
    _add(
        "∠",
        [
            (_line(0.16, 0.78, 0.86, 0.78, 6), HENG),
            (_line(0.16, 0.78, 0.70, 0.22, 6), PIE),
        ],
        0.86,
    )
    _add(
        "⊥",
        [
            (_line(0.50, 0.18, 0.50, 0.72, 6), SHU_CHUILU),
            (_line(0.18, 0.78, 0.82, 0.78, 5), HENG),
        ],
        0.80,
    )
    _add(
        "≡",
        [
            (_line(0.18, 0.32, 0.82, 0.32, 5), HENG),
            (_line(0.18, 0.50, 0.82, 0.50, 5), HENG),
            (_line(0.18, 0.68, 0.82, 0.68, 5), HENG),
        ],
        0.82,
    )
    _add(
        "〔",
        [
            (
                [(0.68, 0.16), (0.36, 0.22), (0.28, 0.50), (0.36, 0.78), (0.68, 0.84)],
                WAN,
            )
        ],
        0.50,
    )
    _add(
        "〕",
        [
            (
                [(0.32, 0.16), (0.64, 0.22), (0.72, 0.50), (0.64, 0.78), (0.32, 0.84)],
                WAN,
            )
        ],
        0.50,
    )
    _add(
        "〖",
        [
            ([(0.78, 0.16), (0.28, 0.16), (0.28, 0.84), (0.78, 0.84)], ZHE),
            ([(0.66, 0.28), (0.40, 0.28), (0.40, 0.72), (0.66, 0.72)], ZHE),
        ],
        0.55,
    )
    _add(
        "〗",
        [
            ([(0.22, 0.16), (0.72, 0.16), (0.72, 0.84), (0.22, 0.84)], ZHE),
            ([(0.34, 0.28), (0.60, 0.28), (0.60, 0.72), (0.34, 0.72)], ZHE),
        ],
        0.55,
    )
    _add("▽", [([(0.50, 0.84), (0.18, 0.18), (0.82, 0.18), (0.50, 0.84)], ZHE)], 0.86)
    _add(
        "▲",
        [
            ([(0.50, 0.16), (0.18, 0.82), (0.82, 0.82), (0.50, 0.16)], ZHE),
            (_line(0.50, 0.28, 0.50, 0.70, 5), SHU_CHUILU),
            (_line(0.32, 0.62, 0.68, 0.62, 5), HENG),
        ],
        0.86,
    )
    _add(
        "▼",
        [
            ([(0.50, 0.84), (0.18, 0.18), (0.82, 0.18), (0.50, 0.84)], ZHE),
            (_line(0.50, 0.30, 0.50, 0.72, 5), SHU_CHUILU),
            (_line(0.32, 0.38, 0.68, 0.38, 5), HENG),
        ],
        0.86,
    )
    _add(
        "◇",
        [([(0.50, 0.14), (0.84, 0.50), (0.50, 0.86), (0.16, 0.50), (0.50, 0.14)], ZHE)],
        0.86,
    )
    _add(
        "◆",
        [
            ([(0.50, 0.14), (0.84, 0.50), (0.50, 0.86), (0.16, 0.50), (0.50, 0.14)], ZHE),
            (_line(0.36, 0.36, 0.64, 0.64, 5), NA),
            (_line(0.64, 0.36, 0.36, 0.64, 5), PIE),
        ],
        0.86,
    )
    _add(
        "℉",
        [
            (_arc(0.18, 0.24, 0.09, 0.09, -90, 270, 12), WAN),
            ([(0.78, 0.16), (0.42, 0.16), (0.42, 0.86)], ZHE),
            (_line(0.42, 0.46, 0.70, 0.46, 5), HENG),
        ],
        0.92,
    )
    _add(
        "‰",
        [
            (_dot(0.22, 0.24, 0.055), DIAN),
            (_line(0.72, 0.16, 0.16, 0.84, 6), PIE),
            (_dot(0.38, 0.76, 0.05), DIAN),
            (_arc(0.70, 0.76, 0.12, 0.12, -90, 270, 10), WAN),
        ],
        0.92,
    )
    _add(
        "㎡",
        [
            (_line(0.10, 0.42, 0.12, 0.84, 5), SHU_CHUILU),
            (
                [(0.12, 0.56), (0.24, 0.42), (0.34, 0.56), (0.34, 0.84)],
                WAN,
            ),
            (
                [(0.34, 0.56), (0.46, 0.42), (0.56, 0.56), (0.56, 0.84)],
                WAN,
            ),
            (
                _arc(0.76, 0.26, 0.10, 0.08, 190, 375, 10) + [(0.84, 0.34), (0.68, 0.48)],
                WAN,
            ),
            (_line(0.66, 0.48, 0.88, 0.48, 4), HENG),
        ],
        0.96,
    )

    missing = [char for char in _NEW if char not in _GLYPHS]
    if missing:
        raise RuntimeError(f"extra_symbols missing definitions: {missing}")
