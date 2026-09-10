"""渲染前字符覆盖扫描。

只分类、归一、报缺字。生僻汉字若盖章字体有字，记入 stamped，不生成骨架。不贴彩色 Emoji。
手机输入的 ``❤️`` 是 ``U+2764`` + ``U+FE0F``，必须在进管线前收成 ``❤``，
否则第二个码位会被当成缺字，整页白渲后失败。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from . import stamp, supplement
from .skeleton import load_bank

VS15 = "\ufe0e"
VS16 = "\ufe0f"
ZWJ = "\u200d"

# 手机/系统常用序列 → 手绘库里的单码位
_ALIASES = {
    "\u2764\ufe0f": "❤",
    "\u2665\ufe0f": "♥",
    "\u2661\ufe0f": "♡",
    "\u2b50": "★",
    "\u2b50\ufe0f": "★",
    "\u2713\ufe0f": "✓",
    "\u2714\ufe0f": "✔",
    "\u2717\ufe0f": "✗",
    "\u301c": "～",
    "〰": "～",
    "＋": "+",
    "＝": "=",
    "＊": "*",
    "／": "/",
    "＼": "\\",
    "｜": "|",
    "＃": "#",
    "＠": "@",
    "＆": "&",
    "＿": "_",
    "＾": "^",
    "｀": "`",
    "＄": "$",
    "＜": "<",
    "＞": ">",
    "−": "-",
    "–": "-",
    "⋯": "…",
    "‥": "…",
    "✕": "×",
    "✖": "×",
    "✘": "✗",
    "☐": "□",
    "☑": "✓",
    "≦": "≤",
    "⩾": "≥",
    "⩽": "≤",
    "≧": "≥",
    **{chr(0xFF10 + i): str(i) for i in range(10)},
    **{chr(0x2170 + i): chr(0x2160 + i) for i in range(10)},
}

_DOODLES = frozenset("♥♡❤★☆✓✔✗")

# CJK 统一表意文字（含扩展区）。𠮷 在 Ext B，不能只写 BMP。
_CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0x2CEB0, 0x2EBEF),
    (0x30000, 0x3134F),
    (0x31350, 0x323AF),
)


@dataclass(frozen=True)
class GlyphHit:
    cluster: str
    category: str  # supported | doodle | rare_hanzi | emoji | other
    codepoint: str


@dataclass(frozen=True)
class CoverageReport:
    supported: tuple[GlyphHit, ...]
    unsupported: tuple[GlyphHit, ...]
    stamped: tuple[GlyphHit, ...] = ()


def codepoint_key(cluster: str) -> str:
    return "+".join(f"U+{ord(ch):04X}" for ch in cluster)


def iter_clusters(text: str) -> list[str]:
    """把 VS16/VS15 粘在前一字上，ZWJ 序列收成一簇。"""
    clusters: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        start = index
        index += 1
        while index < length:
            mark = text[index]
            if mark in (VS15, VS16):
                index += 1
                continue
            if mark == ZWJ and index + 1 < length:
                index += 2
                while index < length and text[index] in (VS15, VS16):
                    index += 1
                continue
            break
        clusters.append(text[start:index])
    return clusters


def _compat_hanzi_alias(cluster: str) -> str:
    """只映射当前会 400、且目标已在骨架/手绘的兼容区汉字。禁止整段 NFKC。"""
    if len(cluster) != 1:
        return cluster
    code = ord(cluster)
    if not (0xF900 <= code <= 0xFAFF):
        return cluster
    if _is_supported(cluster):
        return cluster
    if stamp.enabled() and stamp.has(cluster):
        return cluster
    target = unicodedata.normalize("NFKC", cluster)
    if target != cluster and len(target) == 1 and _is_supported(target):
        return target
    return cluster


def normalize_cluster(cluster: str) -> str:
    mapped = _ALIASES.get(cluster, cluster)
    if len(mapped) > 1 and mapped[-1] in (VS15, VS16):
        mapped = mapped[:-1]
    mapped = _ALIASES.get(mapped, mapped)
    if mapped in (VS15, VS16, ZWJ):
        return ""
    return _compat_hanzi_alias(mapped)


def normalize_text(text: str) -> str:
    return "".join(normalize_cluster(cluster) for cluster in iter_clusters(text))


def _is_cjk_ideograph(char: str) -> bool:
    if len(char) != 1:
        return False
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def _is_emoji(cluster: str) -> bool:
    if ZWJ in cluster:
        return True
    if len(cluster) != 1:
        return any(_is_emoji(ch) for ch in cluster if ch not in (VS15, VS16, ZWJ))
    code = ord(cluster)
    return (
        0x1F300 <= code <= 0x1FAFF
        or 0x1F600 <= code <= 0x1F64F
        or 0x1F900 <= code <= 0x1F9FF
        or 0x2600 <= code <= 0x27BF
        or 0x1F1E6 <= code <= 0x1F1FF
    )


def _is_supported(char: str) -> bool:
    if not char or char.isspace():
        return True
    bank = load_bank()
    return char in bank or supplement.has(char)


def classify_cluster(cluster: str) -> str:
    if not cluster or cluster.isspace():
        return "supported"
    if cluster in _DOODLES:
        return "doodle" if _is_supported(cluster) else "other"
    if _is_supported(cluster):
        return "supported"
    if _is_cjk_ideograph(cluster):
        return "rare_hanzi"
    if _is_emoji(cluster):
        return "emoji"
    return "other"


def scan_text(text: str) -> CoverageReport:
    supported: list[GlyphHit] = []
    unsupported: list[GlyphHit] = []
    stamped: list[GlyphHit] = []
    seen: set[str] = set()
    stamp_on = stamp.enabled()
    for raw in iter_clusters(text):
        cluster = normalize_cluster(raw)
        if not cluster or cluster.isspace():
            continue
        if cluster in seen:
            continue
        seen.add(cluster)
        category = classify_cluster(cluster)
        hit = GlyphHit(cluster=cluster, category=category, codepoint=codepoint_key(cluster))
        if category in ("supported", "doodle"):
            supported.append(hit)
        elif category == "rare_hanzi" and stamp_on and stamp.has(cluster):
            stamped.append(hit)
        else:
            unsupported.append(hit)
    return CoverageReport(
        supported=tuple(supported),
        unsupported=tuple(unsupported),
        stamped=tuple(stamped),
    )


def unsupported_clusters(text: str) -> list[str]:
    return [hit.cluster for hit in scan_text(text).unsupported]


def missing_message(clusters: list[str]) -> str:
    """与 core._missing_message 同一契约，方便 error.js 继续拆字。"""
    shown = "".join(clusters[:20])
    suffix = "..." if len(clusters) > 20 else ""
    return f"font missing {len(clusters)} character(s): {shown}{suffix}"
