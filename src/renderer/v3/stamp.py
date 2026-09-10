"""生僻汉字字体盖章：把已批准 OFL 字形栅格成与 deposit() 相同的墨量场。

禁止 import core / pipeline，避免环依赖。
"""

from __future__ import annotations

import functools
import hashlib
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from renderer.fonts import FontSpec, load_stamp_fonts

from .filters import gaussian_blur_2d
from .legacy import font_stamp_enabled

# 手写体优先，编码补漏最后。只使用 manifest 里 approved 且 commercial_use 的条目。
FONT_ORDER = (
    "kose-xiaolai",
    "yozai",
    "cef-cjk",
    "mashanzheng",
    "lxgw-wenkai",
    "plangothic-p1",
    "plangothic-p2",
)
VARIANT_COUNT = 3

# 用「的」对比 bank 墨迹高度后的字号收缩。字体 em 比骨架字面略满。
SIZE_FIT = 0.90
MASS_GAIN = 0.58
SOFTEN_SIGMA = 0.42
EDGE_JITTER = 0.09

# aspect_mul, slant_add, rot_rad, dx_em, dy_em
_VARIANT = (
    (1.000, 0.000, 0.000, 0.000, 0.000),
    (1.040, 0.014, 0.020, 0.016, -0.010),
    (0.960, -0.012, -0.017, -0.014, 0.012),
)


def enabled() -> bool:
    return font_stamp_enabled()


def reset_caches() -> None:
    _specs.cache_clear()
    _cmap.cache_clear()
    _font_file.cache_clear()


@functools.lru_cache(maxsize=1)
def _specs() -> tuple[FontSpec, ...]:
    by_id = {spec.id: spec for spec in load_stamp_fonts()}
    return tuple(by_id[font_id] for font_id in FONT_ORDER if font_id in by_id)


@functools.lru_cache(maxsize=8)
def _cmap(path: str) -> frozenset[int]:
    from fontTools.ttLib import TTFont

    tt = TTFont(path, fontNumber=0)
    cmap = tt.getBestCmap() or {}
    glyph_order = tt.getGlyphOrder()
    notdef = glyph_order[0] if glyph_order else ".notdef"
    codes = set()
    for codepoint, name in cmap.items():
        if name == notdef:
            continue
        try:
            if tt.getGlyphID(name) == 0:
                continue
        except Exception:
            continue
        codes.add(int(codepoint))
    return frozenset(codes)


def resolve(char: str) -> str | None:
    if len(char) != 1:
        return None
    code = ord(char)
    for spec in _specs():
        if code in _cmap(str(spec.file)):
            return spec.id
    return None


def has(char: str) -> bool:
    return resolve(char) is not None


@functools.lru_cache(maxsize=8)
def _font_file(font_id: str) -> str:
    for spec in _specs():
        if spec.id == font_id:
            return str(spec.file)
    raise KeyError(font_id)


def _stamp_rng(char: str, variant: int, seed: int) -> np.random.Generator:
    material = f"{char}\0{int(variant)}\0{int(seed)}".encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def _age_edges(mass: np.ndarray, rng: np.random.Generator, ss: int) -> np.ndarray:
    """只在轮廓上啃一点，去掉激光切割边。不要再糊一层。"""
    gy, gx = np.gradient(mass)
    edge = np.hypot(gx, gy)
    peak = float(edge.max())
    if peak < 1e-6:
        return mass
    edge = (edge / peak).astype(np.float32)
    if ss >= 2:
        edge = gaussian_blur_2d(edge, 0.35 * ss, pad_mode="constant")
        peak = float(edge.max())
        if peak > 0:
            edge = edge / peak
    noise = rng.standard_normal(mass.shape).astype(np.float32)
    return np.clip(mass + noise * edge * np.float32(EDGE_JITTER), 0.0, None)


def deposit_mass(
    char: str,
    *,
    em_px: float,
    slant: float = 0.0,
    aspect: float = 1.0,
    gravity_dx: float = 0.0,
    gravity_dy: float = 0.0,
    supersample: int = 3,
    mass_gain: float = MASS_GAIN,
    variant: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, tuple[int, int]]:
    """栅格一个字，返回 (墨量场, 相对 em 左上角的目标分辨率偏移)。"""
    font_id = resolve(char)
    if font_id is None:
        raise ValueError(f"stamp font missing {char!r}")

    kind = int(variant) % VARIANT_COUNT
    aspect_mul, slant_add, rot, dx_em, dy_em = _VARIANT[kind]
    used_aspect = max(0.55, float(aspect) * aspect_mul)
    used_slant = float(slant) + slant_add
    used_dx = float(gravity_dx) + dx_em
    used_dy = float(gravity_dy) + dy_em

    ss = max(1, int(supersample))
    scale = float(em_px) * ss
    font_size = max(8, int(round(scale * SIZE_FIT)))
    font = ImageFont.truetype(_font_file(font_id), font_size)
    bbox = font.getbbox(char)
    ink_w = max(1, bbox[2] - bbox[0])
    ink_h = max(1, bbox[3] - bbox[1])

    inner = max(1, int(round(scale)))
    pad = int(round(0.28 * scale))
    shear = math.tan(used_slant)
    extra = (
        int(math.ceil(abs(shear) * (inner + 2 * pad)))
        + int(math.ceil(abs(used_aspect - 1.0) * inner))
        + int(math.ceil(abs(math.sin(rot)) * inner))
        + int(math.ceil(max(abs(used_dx), abs(used_dy)) * scale))
        + 8
    )
    width = inner + 2 * pad + extra
    height = inner + 2 * pad + extra // 2

    em_x0 = pad + extra // 2
    em_y0 = pad + extra // 4
    dx = em_x0 + (inner - ink_w) / 2.0 - bbox[0]
    dy = em_y0 + (inner - ink_h) / 2.0 - bbox[1]

    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).text((dx, dy), char, font=font, fill=255)
    mass = np.asarray(canvas, dtype=np.float32) / 255.0

    cx = width / 2.0
    cy = height / 2.0
    shift_x = used_dx * scale
    shift_y = used_dy * scale
    inv_aspect = 1.0 / used_aspect
    # dest -> source：先还原平移，再还原切变，再还原宽高比
    affine = (
        inv_aspect,
        -shear,
        cx - inv_aspect * (cx + shift_x) + shear * (cy + shift_y),
        0.0,
        1.0,
        -(shift_y),
    )
    warped = Image.fromarray(mass, mode="F").transform(
        (width, height),
        Image.AFFINE,
        affine,
        resample=Image.Resampling.BILINEAR,
    )
    if abs(rot) > 1e-5:
        warped = warped.rotate(
            math.degrees(rot),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )
    mass = np.asarray(warped, dtype=np.float32)
    mass = _age_edges(mass, _stamp_rng(char, kind, seed), ss)

    if SOFTEN_SIGMA > 1e-3:
        mass = gaussian_blur_2d(mass, SOFTEN_SIGMA * ss, pad_mode="constant")
    mass = np.clip(mass * float(mass_gain), 0.0, None).astype(np.float32)

    if ss > 1:
        pad_h = (-height) % ss
        pad_w = (-width) % ss
        if pad_h or pad_w:
            mass = np.pad(mass, ((0, pad_h), (0, pad_w)))
        mass = mass.reshape(
            mass.shape[0] // ss, ss, mass.shape[1] // ss, ss
        ).mean(axis=(1, 3))
        em_x0 = em_x0 / ss
        em_y0 = em_y0 / ss

    rows, cols = np.nonzero(mass > 1e-4)
    if rows.size == 0:
        return np.zeros((1, 1), dtype=np.float32), (0, 0)

    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    cropped = np.ascontiguousarray(mass[y0:y1, x0:x1])
    offset = (int(round(x0 - em_x0)), int(round(y0 - em_y0)))
    return cropped, offset
