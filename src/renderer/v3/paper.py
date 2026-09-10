"""纸张基底与成像仿真。

一张完美干净的渲染图本身就是最大的破绽：真实照片里墨迹和纸面共享同一套
噪声、模糊与压缩统计，而合成图的墨迹"太干净"。这一层负责把整页拉回
真实的成像统计里。
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from .filters import gaussian_blur_2d
from .ink import linear_to_srgb, srgb_to_linear


@dataclass(frozen=True)
class PaperSpec:
    """纸张物理属性。颜色是 sRGB 0-255，尺寸是毫米。"""

    paper_id: str = "office-80g"
    base_rgb: tuple[int, int, int] = (250, 249, 245)
    # 纤维纹理强度（相对反射率）
    fiber_amplitude: float = 0.011
    # 纤维特征尺度（毫米）
    fiber_scale_mm: float = 0.28
    # 大尺度色斑/厚薄不均
    blotch_amplitude: float = 0.008
    blotch_scale_mm: float = 22.0
    # 横线
    ruling: str = "blank"  # blank | lined | grid
    line_pitch_mm: float = 8.0
    line_rgb: tuple[int, int, int] = (176, 196, 220)
    line_width_mm: float = 0.16
    margin_rule_mm: float | None = None
    margin_rule_rgb: tuple[int, int, int] = (222, 158, 158)


@dataclass(frozen=True)
class CaptureSpec:
    """成像链参数。默认值对应"室内自然光下手机随手拍"。"""

    enabled: bool = True
    # 光照场：不均匀度与主方向
    illumination_strength: float = 0.13
    illumination_angle: float = math.radians(35.0)
    # 暗角
    vignette: float = 0.10
    # 镜头轻微散焦（像素）
    defocus_px: float = 0.55
    # 色差（像素）
    chromatic_px: float = 0.22
    # 传感器：光子散粒噪声与读出噪声（相对满量程）
    shot_noise: float = 0.0075
    read_noise: float = 0.0030
    # ISP 锐化
    sharpen: float = 0.28
    # JPEG 质量
    jpeg_quality: int = 90
    # 纸张微曲（页面高度比例的位移幅度）
    curl: float = 0.004


def _value_noise(shape: tuple[int, int], cell: float, rng: np.random.Generator) -> np.ndarray:
    height, width = shape
    gh = max(2, int(math.ceil(height / max(cell, 1.0))) + 1)
    gw = max(2, int(math.ceil(width / max(cell, 1.0))) + 1)
    lattice = rng.standard_normal((gh, gw)).astype(np.float32)
    img = Image.fromarray(lattice, mode='F')
    return np.array(img.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32, copy=True)


def make_paper(
    width: int, height: int, spec: PaperSpec, px_per_mm: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """生成线性光纸面与吸水率图。

    吸水率图在 ``ink.apply_absorbency_bleed`` 里驱动页级洇散：
    纤维密的地方吸得多，墨迹边缘因此毛糙；中性 0.5 不改墨量场。
    """
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    base = srgb_to_linear(np.asarray(spec.base_rgb, dtype=np.float32) / 255.0)
    paper = np.broadcast_to(base, (height, width, 3)).astype(np.float32).copy()

    fiber_cell = max(1.0, spec.fiber_scale_mm * px_per_mm)
    fiber = _value_noise((height, width), fiber_cell, rng)
    fiber += 0.45 * _value_noise((height, width), fiber_cell * 2.7, rng)
    fiber /= np.abs(fiber).max() + 1e-6
    blotch = _value_noise((height, width), max(2.0, spec.blotch_scale_mm * px_per_mm), rng)
    blotch /= np.abs(blotch).max() + 1e-6

    modulation = 1.0 + spec.fiber_amplitude * fiber + spec.blotch_amplitude * blotch
    paper *= modulation[:, :, None]

    _draw_ruling(paper, spec, px_per_mm)

    absorbency = np.clip(0.5 + 0.5 * fiber, 0.0, 1.0).astype(np.float32)
    return np.clip(paper, 0.0, 1.0), absorbency


def _draw_ruling(paper: np.ndarray, spec: PaperSpec, px_per_mm: float) -> None:
    if spec.ruling == "blank" and spec.margin_rule_mm is None:
        return
    height, width, _ = paper.shape
    line_width = max(1.0, spec.line_width_mm * px_per_mm)

    def blend(mask: np.ndarray, rgb: tuple[int, int, int]) -> None:
        color = srgb_to_linear(np.asarray(rgb, dtype=np.float32) / 255.0)
        alpha = mask[:, :, None]
        paper[:] = paper * (1.0 - alpha) + color[None, None, :] * alpha

    if spec.ruling in ("lined", "grid"):
        pitch = spec.line_pitch_mm * px_per_mm
        rows = np.arange(height, dtype=np.float32)
        phase = np.abs(((rows % pitch) + pitch / 2) % pitch - pitch / 2)
        mask = np.clip(1.0 - (phase - line_width / 2) / 0.8, 0.0, 1.0)
        mask = np.broadcast_to(mask[:, None], (height, width))
        blend(mask.astype(np.float32) * 0.85, spec.line_rgb)
    if spec.ruling == "grid":
        pitch = spec.line_pitch_mm * px_per_mm
        cols = np.arange(width, dtype=np.float32)
        phase = np.abs(((cols % pitch) + pitch / 2) % pitch - pitch / 2)
        mask = np.clip(1.0 - (phase - line_width / 2) / 0.8, 0.0, 1.0)
        mask = np.broadcast_to(mask[None, :], (height, width))
        blend(mask.astype(np.float32) * 0.85, spec.line_rgb)
    if spec.margin_rule_mm is not None:
        x = int(spec.margin_rule_mm * px_per_mm)
        if 0 <= x < width:
            mask = np.zeros((paper.shape[0], paper.shape[1]), dtype=np.float32)
            span = max(1, int(round(line_width)))
            mask[:, x : x + span] = 0.8
            blend(mask, spec.margin_rule_rgb)


def _blur(field: np.ndarray, sigma: float) -> np.ndarray:
    """纸面/光照场的模糊。边界按 edge 延拓，避免页边出现暗框。"""
    return gaussian_blur_2d(field, sigma, pad_mode="edge")


def _vshift(source: np.ndarray, rows_down: int) -> np.ndarray:
    """整行垂直位移，边界复制。纯切片，走 memmove 而不是逐元素 gather。"""
    if rows_down == 0:
        return source
    out = np.empty_like(source)
    height = source.shape[0]
    k = int(np.clip(rows_down, -height + 1, height - 1))
    if k > 0:
        out[: height - k] = source[k:]
        out[height - k :] = source[height - 1 : height]
    else:
        out[-k:] = source[: height + k]
        out[:-k] = source[0:1]
    return out


def _apply_curl(image: np.ndarray, curl: float) -> np.ndarray:
    """纸张微曲：每列一个平缓的垂直位移。

    位移只随列变化且幅度不到十个像素，所以按整数位移把列切成几段连续区间，
    每段用切片搬运，再和下移一行的结果做亚像素插值 —— 二维花式索引在整页上
    要 450ms，是这条链最贵的一步，而它一次 gather 都不需要。
    """
    height, width, _ = image.shape
    columns = np.arange(width, dtype=np.float32)
    shift = curl * height * np.sin(np.pi * (columns / max(width - 1, 1)))
    base = np.floor(shift).astype(np.int32)
    frac = (shift - base).astype(np.float32)[None, :, None]

    lower = np.empty_like(image)
    upper = np.empty_like(image)
    edges = [0, *(np.flatnonzero(np.diff(base)) + 1).tolist(), width]
    for start, stop in zip(edges[:-1], edges[1:]):
        step = int(base[start])
        lower[:, start:stop] = _vshift(image[:, start:stop], step)
        upper[:, start:stop] = _vshift(image[:, start:stop], step + 1)
    lower *= 1.0 - frac
    upper *= frac
    lower += upper
    return lower


_SRGB_LUT_SIZE = 4096
_SRGB_LUT = (
    np.clip(
        linear_to_srgb(np.linspace(0.0, 1.0, _SRGB_LUT_SIZE, dtype=np.float64)), 0.0, 1.0
    )
    * 255.0
).round().astype(np.uint8)


def _linear_to_srgb_u8(linear: np.ndarray) -> np.ndarray:
    """线性光 -> sRGB uint8，查表实现。

    ``np.power`` 在整页 1800 万个元素上要 200ms，而结果最终只保留 8 位精度；
    4096 级查表的最大误差不到半个灰阶。
    """
    index = np.clip(linear * (_SRGB_LUT_SIZE - 1), 0, _SRGB_LUT_SIZE - 1).astype(np.uint16)
    return _SRGB_LUT[index]


def _srgb_slope(linear: np.ndarray, *, srgb_of_linear: np.ndarray) -> np.ndarray:
    """d(sRGB)/d(linear)，把线性光量纲的噪声换算到 sRGB 域。

    传输曲线 ``s = 1.055·L^(1/2.4) − 0.055`` 的导数可以写成 ``(s+0.055)/(2.4·L)``，
    于是复用已经算好的 sRGB 值即可 —— 整页再做一次 ``np.power`` 是这条链上最贵
    的一步，而它完全可以省掉。
    """
    safe = np.maximum(linear, np.float32(1e-3))
    return np.clip((srgb_of_linear + np.float32(0.055)) / (np.float32(2.4) * safe), 0.0, 12.0)


def apply_capture(image_linear: np.ndarray, spec: CaptureSpec, seed: int) -> np.ndarray:
    """把线性光图像过一遍相机/扫描链，返回 sRGB uint8。

    顺序严格按物理发生顺序：光照 → 几何 → 光学 → 传感器 → ISP → 压缩。
    """
    if not spec.enabled:
        return (np.clip(linear_to_srgb(image_linear), 0, 1) * 255).astype(np.uint8)

    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    height, width, _ = image_linear.shape

    # 归一化坐标只沿各自的轴变化，用 (H,1) / (1,W) 广播即可。整页展开成
    # (H,W) 的 mgrid 会白白多分配两个几十 MB 的数组 —— A4 页在这里是内存
    # 带宽瓶颈，能不落地的中间量就不要落地。
    columns = np.arange(width, dtype=np.int32)
    rows = np.arange(height, dtype=np.int32)
    nx = (columns.astype(np.float32) / max(width - 1, 1) - 0.5)[None, :]
    ny = (rows.astype(np.float32) / max(height - 1, 1) - 0.5)[:, None]
    radius_sq = nx**2 + ny**2

    # 光照场（方向梯度 + 二次项）与暗角都是低频乘性场，合成一个再乘一次整页。
    # radius**2.2 == (radius²)**1.1，省掉整页的 sqrt。
    direction = math.cos(spec.illumination_angle) * nx + math.sin(spec.illumination_angle) * ny
    shade = 1.0 + spec.illumination_strength * (direction - 0.35 * radius_sq)
    if spec.vignette > 1e-4:
        shade = shade * (1.0 - spec.vignette * (radius_sq / 0.7071**2) ** 1.1)
    out = image_linear * shade[:, :, None].astype(np.float32)

    if spec.curl > 1e-5:
        out = _apply_curl(out, spec.curl)

    # 光学：镜头散焦。必须发生在传感器之前 —— 先加噪声再模糊会被高斯核吃掉
    # 大半高频能量，纸面因此比真实照片干净一倍，成为合成检测的头号破绽。
    np.clip(out, 0.0, 1.0, out=out)
    result = _linear_to_srgb_u8(out)
    if spec.defocus_px > 1e-3:
        result = np.asarray(
            Image.fromarray(result).filter(
                ImageFilter.GaussianBlur(radius=float(spec.defocus_px))
            )
        )

    # 传感器：光子散粒噪声（与信号强度相关）+ 读出噪声。噪声本身是线性光量纲，
    # 用 sRGB 传输曲线的斜率换算过来，省掉一次整页的来回转换。单通道噪声广播到
    # 三通道：色度噪声随后会被 JPEG 的 4:2:0 下采样吃掉，算三份是白算。
    if spec.shot_noise > 1e-5 or spec.read_noise > 1e-5:
        gain = np.sqrt(out, dtype=np.float32)
        gain *= spec.shot_noise
        gain += spec.read_noise
        gain *= _srgb_slope(out, srgb_of_linear=result.astype(np.float32) / np.float32(255.0))
        gain *= np.float32(255.0)
        gain *= rng.standard_normal((height, width, 1), dtype=np.float32)
        noisy = result.astype(np.float32)
        noisy += gain
        result = np.clip(noisy, 0.0, 255.0).astype(np.uint8)

    if spec.sharpen > 1e-3:
        result = np.asarray(
            Image.fromarray(result).filter(
                ImageFilter.UnsharpMask(
                    radius=1.1, percent=int(round(spec.sharpen * 100)), threshold=0
                )
            )
        )

    # JPEG：块效应与色度下采样是照片的强特征
    if 1 <= spec.jpeg_quality <= 100:
        buffer = io.BytesIO()
        Image.fromarray(result).save(buffer, format="JPEG", quality=spec.jpeg_quality, subsampling=2)
        buffer.seek(0)
        result = np.asarray(Image.open(buffer).convert("RGB"))
    return result
