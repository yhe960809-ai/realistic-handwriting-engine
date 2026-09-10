"""写错再涂改：默认关闭。count=0 必须返回同一字符串对象。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .coverage import _is_cjk_ideograph
from .skeleton import GlyphSkeleton, load_bank
from .strokes import HENG, NA, PIE, ZHE, profile_for
from .supplement import _line

STYLES = frozenset({"strike", "scribble", "cross"})
MAX_COUNT = 8
_RNG_XOR = 0xA5A5C3C3

_LOOKALIKES = {
    "的": "地",
    "地": "的",
    "得": "的",
    "在": "再",
    "再": "在",
    "未": "末",
    "末": "未",
    "己": "已",
    "已": "己",
    "人": "入",
    "入": "人",
    "土": "士",
    "士": "土",
    "天": "夭",
    "夭": "天",
    "日": "目",
    "目": "日",
    "干": "千",
    "千": "干",
    "大": "太",
    "太": "大",
    "木": "本",
    "本": "木",
    "问": "间",
    "间": "问",
}
_COMMON = "的是不了在有人这中大为上个国我以要他时来"


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng((int(seed) ^ _RNG_XOR) & 0xFFFFFFFF)


def is_eligible(char: str) -> bool:
    if not _is_cjk_ideograph(char):
        return False
    return char in load_bank()


def pick_wrong(char: str, rng: np.random.Generator) -> str | None:
    bank = load_bank()
    look = _LOOKALIKES.get(char)
    if look and look != char and look in bank:
        return look
    choices = [item for item in _COMMON if item != char and item in bank]
    if not choices:
        return None
    return str(choices[int(rng.integers(0, len(choices)))])


def expand(text: str, count: int, seed: int) -> tuple[str, frozenset[int]]:
    """在合格汉字前插入错字。count<=0 时 new_text is text。"""
    if count <= 0:
        return text, frozenset()
    eligible = [index for index, char in enumerate(text) if is_eligible(char)]
    n = min(int(count), MAX_COUNT, len(eligible))
    if n <= 0:
        return text, frozenset()
    rng = _rng(seed)
    picks = sorted((int(x) for x in rng.choice(eligible, size=n, replace=False)), reverse=True)
    out = text
    scribbled: list[int] = []
    for index in picks:
        wrong = pick_wrong(out[index], rng)
        if wrong is None:
            continue
        out = out[:index] + wrong + out[index:]
        scribbled = [mark + 1 if mark >= index else mark for mark in scribbled]
        scribbled.append(index)
    if not scribbled:
        return text, frozenset()
    return out, frozenset(scribbled)


def expand_blocks(
    texts: list[str],
    count: int,
    seed: int,
    *,
    fits: Callable[[int, str], bool] | None = None,
) -> tuple[list[str], list[frozenset[int]]]:
    """多块共用次数预算。fits(block_index, new_text) 为假则撤销该点。"""
    if count <= 0:
        return texts, [frozenset() for _ in texts]
    eligible: list[tuple[int, int]] = []
    for block_i, text in enumerate(texts):
        for local_i, char in enumerate(text):
            if is_eligible(char):
                eligible.append((block_i, local_i))
    n = min(int(count), MAX_COUNT, len(eligible))
    if n <= 0:
        return texts, [frozenset() for _ in texts]
    rng = _rng(seed)
    picks = [eligible[int(i)] for i in rng.choice(len(eligible), size=n, replace=False)]
    picks.sort(key=lambda item: (item[0], item[1]), reverse=True)
    out = list(texts)
    marks: list[list[int]] = [[] for _ in texts]
    for block_i, local_i in picks:
        current = out[block_i]
        wrong = pick_wrong(current[local_i], rng)
        if wrong is None:
            continue
        candidate = current[:local_i] + wrong + current[local_i:]
        if fits is not None and not fits(block_i, candidate):
            continue
        out[block_i] = candidate
        marks[block_i] = [mark + 1 if mark >= local_i else mark for mark in marks[block_i]]
        marks[block_i].append(local_i)
    if not any(marks):
        return texts, [frozenset() for _ in texts]
    return out, [frozenset(item) for item in marks]


def scratch_strokes(style: str) -> list[tuple[list[tuple[float, float]], int]]:
    kind = style if style in STYLES else "strike"
    if kind == "scribble":
        return [
            (
                [
                    (0.12, 0.28),
                    (0.88, 0.36),
                    (0.14, 0.50),
                    (0.86, 0.62),
                    (0.16, 0.78),
                ],
                ZHE,
            )
        ]
    if kind == "cross":
        return [
            (_line(0.16, 0.20, 0.86, 0.82, 6), NA),
            (_line(0.86, 0.20, 0.16, 0.82, 6), PIE),
        ]
    return [
        (_line(0.08, 0.42, 0.92, 0.42, 6), HENG),
        (_line(0.10, 0.58, 0.90, 0.58, 6), HENG),
    ]


def scratch_skeleton(style: str) -> GlyphSkeleton:
    strokes = scratch_strokes(style)
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
        xs = np.interp(targets, cumulative, pts[:, 0])
        ys = np.interp(targets, cumulative, pts[:, 1])
        resampled.append(np.stack([xs, ys], axis=1).astype(np.float32))
        type_ids.append(type_id)
        lengths.append(total)
        weights.append(profile_for(type_id).ink_weight)
    points = np.stack(resampled, axis=0)
    merged = points.reshape(-1, 2)
    key = f"\x00scratch:{style if style in STYLES else 'strike'}"
    return GlyphSkeleton(
        char=key,
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
