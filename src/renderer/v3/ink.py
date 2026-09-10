"""墨迹沉积与光学合成。

累加的是**墨量（mass）**而不是 alpha —— 这是物理正确的做法，
笔画交叉处会自然更黑，慢笔和转折处会自然堆积。
最后用 Beer-Lambert 分通道吸收转成颜色，所以薄墨处不是灰，
而是向墨水色相偏移（蓝黑墨薄处偏蓝），这是肉眼可感的真实细节。
"""

from __future__ import annotations

import math

import numpy as np

from .filters import derivative, gaussian_blur_2d, median, quantile
from .style import Pen
from .trajectory import PenPath

# 边缘抗锯齿过渡带（超采样像素）
_EDGE_SOFTNESS = 0.9


def _resample_path(path: PenPath, spacing_px: float, scale: float) -> dict[str, np.ndarray]:
    """把一条笔画重采样到指定的像素间距，所有并行量一起插值。"""
    xy = path.xy * scale
    deltas = np.diff(xy, axis=0)
    seg = np.hypot(deltas[:, 0], deltas[:, 1])
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cumulative[-1])
    if total <= 1e-6:
        count = 1
        targets = np.zeros(1)
    else:
        count = max(2, int(math.ceil(total / max(spacing_px, 1e-3))) + 1)
        targets = np.linspace(0.0, total, count)
    return {
        "x": np.interp(targets, cumulative, xy[:, 0]),
        "y": np.interp(targets, cumulative, xy[:, 1]),
        "pressure": np.interp(targets, cumulative, path.pressure),
        "width_scale": np.interp(targets, cumulative, path.width_scale),
        "speed": np.interp(targets, cumulative, path.speed),
        "arc": targets,
        "total": np.asarray([total]),
    }


def _curvature_along(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return np.zeros(len(x))
    dx = derivative(x)
    dy = derivative(y)
    ddx = derivative(dx)
    ddy = derivative(dy)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = np.power(dx**2 + dy**2, 1.5) + 1e-12
    return numerator / denominator


def _starvation(
    arc: np.ndarray, pressure: np.ndarray, speed: np.ndarray, pen: Pen, rng: np.random.Generator
) -> np.ndarray:
    """有限供墨模型。断墨是路程与速度驱动的相关过程，不是独立随机孔洞。"""
    if pen.starvation <= 1e-4:
        return np.zeros_like(arc)
    steps = np.diff(arc, prepend=arc[0])
    speed_norm = speed / (median(speed) + 1e-9)
    reservoir = 1.0
    out = np.empty_like(arc)
    demand = steps * (0.32 + 0.42 * pressure) * (0.7 + 0.5 * np.clip(speed_norm, 0.0, 3.0))
    recovery = pen.reservoir_recovery * steps * 0.02
    for index in range(len(arc)):
        reservoir = min(pen.reservoir, reservoir + recovery[index])
        reservoir = max(0.0, reservoir - pen.starvation * demand[index] * 0.05)
        out[index] = 1.0 - reservoir / pen.reservoir
    # 叠加低频供墨波动，产生成片的飞白而不是均匀变淡
    phase = rng.random() * math.tau
    frequency = 0.35 + rng.random() * 0.5
    wave = np.clip(np.sin(arc * frequency + phase) - 0.55, 0.0, None) / 0.45
    return np.clip(pen.starvation * (out * 2.2 + wave * 0.8), 0.0, 0.85)


_SPLAT_CHUNK = 40


def _splat(
    canvas: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radius: np.ndarray,
    amplitude: np.ndarray,
    *,
    nib_aspect: float,
    nib_azimuth: float,
) -> None:
    """把一串笔尖足迹累加进墨量场。

    用 bincount 做散射累加：np.add.at 在这个规模上慢一个量级。
    分块 + float32，避免一次分配整笔的 [S,K,K] 大数组把缓存冲掉。
    """
    height, width = canvas.shape
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    radius = np.asarray(radius, dtype=np.float32)
    amplitude = np.asarray(amplitude, dtype=np.float32)
    max_radius = float(radius.max())
    half = int(math.ceil(max_radius + _EDGE_SOFTNESS)) + 1
    offsets = np.arange(-half, half + 1, dtype=np.int32)
    elliptical = abs(nib_aspect - 1.0) > 1e-3
    if elliptical:
        cos_a = np.float32(math.cos(nib_azimuth))
        sin_a = np.float32(math.sin(nib_azimuth))
        aspect = np.float32(nib_aspect)
    softness = np.float32(_EDGE_SOFTNESS)

    for start in range(0, len(x), _SPLAT_CHUNK):
        sl = slice(start, start + _SPLAT_CHUNK)
        xs, ys, rs, amps = x[sl], y[sl], radius[sl], amplitude[sl]
        base_x = np.floor(xs).astype(np.int32)
        base_y = np.floor(ys).astype(np.int32)
        grid_x = base_x[:, None] + offsets[None, :]
        grid_y = base_y[:, None] + offsets[None, :]
        dx = grid_x.astype(np.float32) - xs[:, None]
        dy = grid_y.astype(np.float32) - ys[:, None]

        if elliptical:
            rx = dx[:, None, :] * cos_a + dy[:, :, None] * sin_a
            ry = -dx[:, None, :] * sin_a + dy[:, :, None] * cos_a
            distance = np.sqrt((rx / aspect) ** 2 + ry**2, dtype=np.float32)
        else:
            distance = np.sqrt(dx[:, None, :] ** 2 + dy[:, :, None] ** 2, dtype=np.float32)

        coverage = np.clip((rs[:, None, None] - distance) / softness + 0.5, 0.0, 1.0)
        contribution = coverage * amps[:, None, None]
        valid_x = (grid_x >= 0) & (grid_x < width)
        valid_y = (grid_y >= 0) & (grid_y < height)
        mask = valid_y[:, :, None] & valid_x[:, None, :]
        if not mask.any():
            continue
        flat_index = (
            np.clip(grid_y, 0, height - 1)[:, :, None] * width
            + np.clip(grid_x, 0, width - 1)[:, None, :]
        )
        weights = np.where(mask, contribution, np.float32(0.0))
        canvas.reshape(-1)[:] += np.bincount(
            flat_index.ravel(), weights=weights.ravel(), minlength=height * width
        )


def _gaussian_blur(field: np.ndarray, sigma: float) -> np.ndarray:
    """可分离高斯模糊。用于纸纤维洇散，保持总墨量守恒。

    边界补零：墨量场是"多少墨"，纸面外没有墨，补零才守恒。
    """
    return gaussian_blur_2d(field, sigma, pad_mode="constant")


def deposit(
    paths: list[PenPath],
    *,
    em_px: float,
    pen: Pen,
    px_per_mm: float,
    supersample: int,
    seed: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """把一个字的所有笔画沉积成墨量场。

    返回 (墨量场, (左上角相对字身原点的像素偏移))，已降采样到目标分辨率。
    """
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    scale = em_px * supersample

    nib_radius = 0.5 * pen.nib_width_mm * px_per_mm * supersample
    bleed = pen.bleed_sigma_px * supersample * max(px_per_mm / 10.0, 0.4)
    # 画布必须容得下笔宽、洇散、以及骨架越界的笔画（撇捺常常出框）
    margin = int(math.ceil(nib_radius * 2.6 + bleed * 3.0 + supersample * 2))

    all_xy = np.concatenate([path.xy for path in paths], axis=0) * scale
    min_x = math.floor(all_xy[:, 0].min()) - margin
    min_y = math.floor(all_xy[:, 1].min()) - margin
    max_x = math.ceil(all_xy[:, 0].max()) + margin
    max_y = math.ceil(all_xy[:, 1].max()) + margin
    width = int(max_x - min_x)
    height = int(max_y - min_y)
    if width <= 0 or height <= 0:
        return np.zeros((1, 1), dtype=np.float32), (0, 0)

    canvas = np.zeros((height, width), dtype=np.float32)
    spacing = max(0.85, nib_radius * 0.50)

    for path in paths:
        sampled = _resample_path(path, spacing, scale)
        x = sampled["x"] - min_x
        y = sampled["y"] - min_y
        if len(x) < 2:
            continue
        pressure = sampled["pressure"]
        arc = sampled["arc"]
        speed = sampled["speed"]
        curvature = _curvature_along(x, y)
        curvature_norm = curvature / (quantile(curvature, 0.88) + 1e-9)
        speed_norm = speed / (median(speed) + 1e-9)

        # 半径：赫兹接触 r ∝ P^(1/3)，再乘笔型宽度剖面与速度损失
        radius = (
            nib_radius
            * sampled["width_scale"]
            * np.power(1.0 + pen.pressure_width_gain * pressure, 1.0 / 3.0)
            * (1.0 - pen.speed_width_loss * np.clip(speed_norm - 1.0, -0.6, 1.6))
        )
        radius = np.clip(radius, 0.35 * supersample, nib_radius * 3.2)

        # 流量：压力正相关，断墨负相关，慢笔与转折处堆积
        starvation = _starvation(arc, pressure, speed, pen, rng)
        pooling = pen.pooling * (
            0.55 * np.clip(1.0 - speed_norm, 0.0, 1.0) + 0.85 * np.clip(curvature_norm, 0.0, 1.4)
        )
        flow = (
            pen.base_flow
            * (1.0 + pen.pressure_flow_gain * pressure)
            * (1.0 - starvation)
            * (1.0 + pooling)
        )
        if path.is_ligature:
            flow *= 0.55

        # 振幅守恒：让沉积后的面密度等于 flow，与采样密度和笔宽无关
        steps = np.diff(arc, prepend=arc[0] - spacing)
        amplitude = flow * steps / np.maximum(2.0 * radius, 1e-6)

        _splat(
            canvas,
            x,
            y,
            radius,
            amplitude,
            nib_aspect=pen.nib_aspect,
            nib_azimuth=pen.nib_azimuth,
        )

    # 纸纤维洇散：墨从落点向外渗，总墨量守恒
    if bleed > 1e-3:
        spread = _gaussian_blur(canvas, bleed)
        canvas = canvas * 0.35 + spread * 0.65

    # 咖啡环：干燥时墨向边界迁移，边缘比中心深
    if pen.edge_ring > 1e-3:
        smooth = _gaussian_blur(canvas, max(bleed, 0.8 * supersample))
        gradient_y, gradient_x = np.gradient(smooth)
        edge = np.hypot(gradient_x, gradient_y)
        peak = quantile(edge, 0.995) + 1e-9
        canvas = canvas + pen.edge_ring * np.clip(edge / peak, 0.0, 1.5) * np.clip(canvas, 0.0, 2.0)

    # 降采样到目标分辨率（面积平均）
    if supersample > 1:
        pad_h = (-height) % supersample
        pad_w = (-width) % supersample
        if pad_h or pad_w:
            canvas = np.pad(canvas, ((0, pad_h), (0, pad_w)))
        canvas = canvas.reshape(
            canvas.shape[0] // supersample, supersample, canvas.shape[1] // supersample, supersample
        ).mean(axis=(1, 3))

    offset = (int(min_x // supersample), int(min_y // supersample))
    return canvas.astype(np.float32), offset


# 吸水偏离 0.5 时，把这一比例的墨从原场抽到模糊场（或反向）。
# 0.5 中性纸面时整页恒等，避免在笔端洇散上再叠一层全局模糊。
_ABSORBENCY_STRENGTH = 0.7


def apply_absorbency_bleed(
    mass: np.ndarray,
    absorbency: np.ndarray,
    *,
    sigma: float,
    strength: float = _ABSORBENCY_STRENGTH,
) -> np.ndarray:
    """用纸张吸水率做空间变化洇散。

    ``absorbency=0.5`` 时与输入逐位相同。大于 0.5 的区域多洇，小于 0.5 少洇。
    纤维密处边缘变毛，匀质纸不会被再糊一层。
    """
    if sigma <= 1e-3 or strength <= 1e-4:
        return mass
    if absorbency.shape != mass.shape:
        raise ValueError(
            f"absorbency shape {absorbency.shape} != mass shape {mass.shape}"
        )
    spread = gaussian_blur_2d(mass, float(sigma), pad_mode="constant")
    drive = (absorbency.astype(np.float32, copy=False) - np.float32(0.5)) * np.float32(
        2.0 * strength
    )
    return np.clip(mass + drive * (spread - mass), 0.0, None)


def composite(
    paper_linear: np.ndarray, mass: np.ndarray, pen: Pen, *, density_gain: float = 2.35
) -> np.ndarray:
    """Beer-Lambert 分通道吸收，在线性光空间合成。

    ``paper_linear`` 是 [H, W, 3] 的线性光纸面，``mass`` 是同尺寸墨量场。
    """
    absorb = np.asarray(pen.absorb_rgb, dtype=np.float32)
    ink_linear = srgb_to_linear(np.asarray(pen.ink_rgb, dtype=np.float32) / 255.0)
    optical = np.clip(mass, 0.0, None)[:, :, None] * absorb[None, None, :] * density_gain
    transmittance = np.exp(-optical)
    return paper_linear * transmittance + ink_linear[None, None, :] * (1.0 - transmittance)


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return np.where(value <= 0.0031308, value * 12.92, 1.055 * np.power(value, 1 / 2.4) - 0.055)
