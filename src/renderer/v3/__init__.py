"""V3 真迹引擎：骨架 → 风格 → 轨迹 → 墨迹 → 纸面 → 成像。"""

from .ink import apply_absorbency_bleed, composite, deposit, linear_to_srgb, srgb_to_linear
from .skeleton import GlyphSkeleton, SkeletonBank, load_bank
from .style import PENS, PRESETS, Pen, WriterStyle, get_style, list_styles
from .trajectory import PenPath, synthesize

__all__ = [
    "GlyphSkeleton",
    "PENS",
    "PRESETS",
    "Pen",
    "PenPath",
    "SkeletonBank",
    "WriterStyle",
    "apply_absorbency_bleed",
    "composite",
    "deposit",
    "get_style",
    "linear_to_srgb",
    "list_styles",
    "load_bank",
    "srgb_to_linear",
    "synthesize",
]
