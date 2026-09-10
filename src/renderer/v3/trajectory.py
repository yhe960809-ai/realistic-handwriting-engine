"""轨迹合成：把静态骨架变成一次真实的书写动作。

这一层是 V3 与"字体+抖动"的分水岭。产出是逐点的
(x, y, 时间, 压力, 宽度倍率)，后续墨迹层完全由它驱动。

四个关键机制
------------
1. 结构变形  —— 中宫、部件比例、重心、倾斜、横画上扬，按书写风格施加。
2. MLS 相似变形 —— 每一次出现独立的弹性形变。用相似变换而非随机抖动，
   形变后的字仍然是合法的字，只是"这一次写得瘦一点、右下角塌一点"。
3. Σ-Lognormal 速度剖面 + 2/3 幂律 —— 人体快速运动的速度必然呈对数正态钟形，
   且曲率越大速度越低。有了它，转折才会自然加粗、收笔才会自然出锋。
4. 牵丝连笔 —— 笔画间距足够近且书写够快时生成低压力连接弧。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .filters import derivative_rows, quantile
from .legacy import USE_LEGACY_INK
from .skeleton import GlyphSkeleton
from .strokes import HENG, evaluate_width_profile, profile_for
from .style import WriterStyle


@dataclass(frozen=True)
class PenPath:
    """一笔（或一段牵丝）的完整书写动作。坐标在字身归一化空间 [0,1]。"""

    xy: np.ndarray  # [M, 2]
    time_s: np.ndarray  # [M]
    pressure: np.ndarray  # [M] 0~1
    width_scale: np.ndarray  # [M] 相对标称笔宽
    speed: np.ndarray  # [M] em/s
    type_id: int
    is_ligature: bool


def _rng(seed: int, *parts: object) -> np.random.Generator:
    payload = "\x1f".join(str(part) for part in (seed, *parts)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def _arc_params(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """返回 (归一化累计弧长, 总弧长)。"""
    deltas = np.diff(xy, axis=0)
    seg = np.hypot(deltas[:, 0], deltas[:, 1])
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cumulative[-1])
    if total <= 1e-9:
        return np.zeros(len(xy)), 0.0
    return cumulative / total, total


def _arc_length(xy: np.ndarray) -> float:
    """只要总弧长时用它：省掉 _arc_params 的归一化除法与 fractions 数组。

    刻意用 ``cumsum[-1]`` 而不是 ``sum()``：两者的浮点累加顺序不同，结果可能差
    一个 ULP，而这个值会经 ``int()`` 决定采样点数 —— 边界上翻一格就是另一张图。
    保持与 :func:`_arc_params` 逐位一致，这次改动才是纯粹的提速。
    """
    deltas = np.diff(xy, axis=0)
    seg = np.hypot(deltas[:, 0], deltas[:, 1])
    if seg.size == 0:
        return 0.0
    total = float(np.cumsum(seg)[-1])
    return 0.0 if total <= 1e-9 else total


def _resample_uniform(xy: np.ndarray, count: int) -> np.ndarray:
    fractions, total = _arc_params(xy)
    if total <= 1e-9:
        return np.repeat(xy[:1], count, axis=0)
    targets = np.linspace(0.0, 1.0, count)
    x = np.interp(targets, fractions, xy[:, 0])
    y = np.interp(targets, fractions, xy[:, 1])
    return np.stack([x, y], axis=1)


def _smooth_path(xy: np.ndarray, sigma_samples: float) -> np.ndarray:
    """沿路径高斯平滑 —— 圆化转角，模拟快写时的圆转用笔。

    端点用边界复制，避免笔画两端被拉短。
    """
    if sigma_samples <= 0.35 or len(xy) < 5:
        return xy
    radius = max(1, int(math.ceil(sigma_samples * 2.5)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(xy, ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.empty_like(xy)
    for axis in (0, 1):
        smoothed[:, axis] = np.convolve(padded[:, axis], kernel, mode="valid")
    return smoothed


def _curvature(xy: np.ndarray) -> np.ndarray:
    """离散曲率（1/半径），单位与坐标一致。"""
    if len(xy) < 3:
        return np.zeros(len(xy))
    first = derivative_rows(xy)
    second = derivative_rows(first)
    numerator = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    denominator = np.power(first[:, 0] ** 2 + first[:, 1] ** 2, 1.5) + 1e-12
    return numerator / denominator


def _normals(xy: np.ndarray) -> np.ndarray:
    tangent = derivative_rows(xy)
    norm = np.hypot(tangent[:, 0], tangent[:, 1]) + 1e-12
    tangent = tangent / norm[:, None]
    return np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)


# ------------------------------------------------------------------ 结构变形

def _apply_structure(
    strokes: list[np.ndarray], skeleton: GlyphSkeleton, style: WriterStyle, rng: np.random.Generator
) -> list[np.ndarray]:
    structure = style.structure
    center = np.array([0.5, 0.5])

    # 部件级：左右/上下结构的比例偏置与相对位移
    groups = skeleton.structure.get("groups") or []
    kind = skeleton.structure.get("kind", "single")
    if kind in ("lr", "tb") and len(groups) == 2:
        axis = 0 if kind == "lr" else 1
        split = float(skeleton.structure.get("split", 0.5))
        jitter = structure.component_jitter
        for group_index, indices in enumerate(groups):
            if not indices:
                continue
            # 第一个部件按 component_ratio 放大/缩小，第二个反向补偿
            sign = 1.0 if group_index == 0 else -1.0
            scale = 1.0 + sign * structure.component_ratio
            anchor = 0.0 if group_index == 0 else 1.0
            pivot = split if group_index == 0 else split
            offset = rng.normal(0.0, jitter, size=2)
            for index in indices:
                pts = strokes[index]
                pts[:, axis] = pivot + (pts[:, axis] - pivot) * scale
                # 靠外的一侧保持贴边，避免整字被推出字框
                pts[:, axis] += (anchor - pivot) * (1.0 - scale) * 0.35
                pts += offset

    merged = np.concatenate(strokes, axis=0)
    span = np.maximum(merged.max(axis=0) - merged.min(axis=0), 1e-6)
    glyph_center = merged.min(axis=0) + span * 0.5
    radius = float(np.hypot(*(span * 0.5))) + 1e-6

    out: list[np.ndarray] = []
    for index, pts in enumerate(strokes):
        pts = pts.copy()

        # 中宫松紧：靠近字心的部分按 zhonggong 收放，靠边的保持原位
        offset = pts - glyph_center
        distance = np.hypot(offset[:, 0], offset[:, 1]) / radius
        factor = structure.zhonggong + (1.0 - structure.zhonggong) * np.clip(distance, 0.0, 1.0)
        pts = glyph_center + offset * factor[:, None]

        # 字面宽高比
        aspect = math.sqrt(max(structure.aspect, 1e-3))
        pts[:, 0] = glyph_center[0] + (pts[:, 0] - glyph_center[0]) * aspect
        pts[:, 1] = glyph_center[1] + (pts[:, 1] - glyph_center[1]) / aspect

        # 横画上扬：绕起笔点抬起横向笔画。印刷体没有这个特征，手写普遍有。
        if int(skeleton.type_ids[index]) == HENG and structure.heng_rise != 0.0:
            angle = -structure.heng_rise
            pivot = pts[0].copy()
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            rel = pts - pivot
            pts = pivot + np.stack(
                [rel[:, 0] * cos_a - rel[:, 1] * sin_a, rel[:, 0] * sin_a + rel[:, 1] * cos_a],
                axis=1,
            )

        out.append(pts)

    # 整字倾斜（切变）+ 重心偏移
    if structure.slant or structure.gravity_dx or structure.gravity_dy:
        shear = math.tan(structure.slant)
        for pts in out:
            pts[:, 0] += shear * (glyph_center[1] - pts[:, 1])
            pts[:, 0] += structure.gravity_dx
            pts[:, 1] += structure.gravity_dy
    _ = center
    return out


# ------------------------------------------------------------------ MLS 形变

def _mls_similarity(
    points: np.ndarray, source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Moving Least Squares 相似变形。

    相似变换在复数域是 f(z) = a·z + b，加权最小二乘有闭式解，
    因此整字所有点可以一次性向量化求解。
    """
    v = points[:, 0] + 1j * points[:, 1]
    p = source[:, 0] + 1j * source[:, 1]
    q = target[:, 0] + 1j * target[:, 1]

    distance_sq = np.abs(v[:, None] - p[None, :]) ** 2
    weights = 1.0 / (distance_sq + 1e-8)
    # 落在控制点上的查询点权重会爆掉，直接钳到有限值即可
    weights = np.minimum(weights, 1e8)
    weight_sum = weights.sum(axis=1)

    p_star = (weights @ p) / weight_sum
    q_star = (weights @ q) / weight_sum
    p_hat = p[None, :] - p_star[:, None]
    q_hat = q[None, :] - q_star[:, None]

    numerator = (weights * np.conj(p_hat) * q_hat).sum(axis=1)
    denominator = (weights * (np.abs(p_hat) ** 2)).sum(axis=1) + 1e-12
    a = numerator / denominator
    result = q_star + a * (v - p_star)
    return np.stack([result.real, result.imag], axis=1)


def _elastic_deform(
    strokes: list[np.ndarray], style: WriterStyle, rng: np.random.Generator
) -> list[np.ndarray]:
    amplitude = style.structure.elastic
    if amplitude <= 1e-4:
        return strokes
    grid = np.linspace(-0.15, 1.15, 4)
    xx, yy = np.meshgrid(grid, grid)
    source = np.stack([xx.ravel(), yy.ravel()], axis=1)
    # 边界控制点扰动减半：整字轮廓不能塌，塌了就不像字了
    edge = (
        (source[:, 0] <= grid[0] + 1e-6)
        | (source[:, 0] >= grid[-1] - 1e-6)
        | (source[:, 1] <= grid[0] + 1e-6)
        | (source[:, 1] >= grid[-1] - 1e-6)
    )
    scale = np.where(edge, 0.45, 1.0)[:, None]
    target = source + rng.normal(0.0, amplitude, size=source.shape) * scale

    lengths = [len(pts) for pts in strokes]
    merged = np.concatenate(strokes, axis=0)
    warped = _mls_similarity(merged, source, target)
    out: list[np.ndarray] = []
    cursor = 0
    for length in lengths:
        out.append(warped[cursor : cursor + length])
        cursor += length
    return out


# ------------------------------------------------------------------ 动力学

def _lognormal_speed(
    fractions: np.ndarray, submovements: np.ndarray, mu: float, sigma: float
) -> np.ndarray:
    """Σ-Lognormal：若干条对数正态速度脉冲的叠加。

    每个子运动在其起点之后产生一条右偏钟形速度曲线，
    这是 Plamondon 快速人体运动动力学理论给出的形状。
    """
    speed = np.zeros_like(fractions)
    for start, extent, gain in submovements:
        rel = (fractions - start) / max(extent, 1e-3)
        valid = rel > 1e-4
        if not valid.any():
            continue
        value = np.zeros_like(fractions)
        log_rel = np.log(rel[valid])
        value[valid] = np.exp(-((log_rel - mu) ** 2) / (2.0 * sigma**2)) / (
            rel[valid] * sigma * math.sqrt(2.0 * math.pi)
        )
        speed += gain * value
    peak = speed.max()
    if peak <= 1e-9:
        return np.ones_like(fractions)
    return speed / peak


def _submovements(fractions: np.ndarray, curvature: np.ndarray) -> np.ndarray:
    """按曲率极大值切分子运动。转折处是一次新的加速。"""
    boundaries = [0.0]
    if len(curvature) > 6:
        threshold = quantile(curvature, 0.82)
        window = max(2, len(curvature) // 12)
        for index in range(window, len(curvature) - window):
            local = curvature[index - window : index + window + 1]
            if curvature[index] >= threshold and curvature[index] >= local.max() - 1e-12:
                position = float(fractions[index])
                if position - boundaries[-1] > 0.18:
                    boundaries.append(position)
    boundaries.append(1.0)
    result = []
    for index in range(len(boundaries) - 1):
        start = boundaries[index]
        extent = max(boundaries[index + 1] - start, 0.12)
        # 后续子运动的幅度略小：手在一笔之内是逐渐减速的
        result.append((start - extent * 0.12, extent, 1.0 / (1.0 + 0.35 * index)))
    return np.asarray(result, dtype=np.float64)


def _ou_noise(count: int, sigma: float, rho: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0.0 or count <= 0:
        return np.zeros(count)
    innovation = rng.normal(0.0, sigma * math.sqrt(1.0 - rho**2), size=count)
    y0 = rng.normal(0.0, sigma)
    kernel = np.power(rho, np.arange(count, dtype=np.float64))
    conv = np.convolve(innovation, kernel)[:count]
    return y0 * rho * kernel + conv


def _build_stroke_path(
    xy: np.ndarray,
    type_id: int,
    style: WriterStyle,
    rng: np.random.Generator,
    start_time: float,
    em_px: float,
) -> PenPath:
    profile = profile_for(type_id)
    motion = style.motion
    brush = style.brush

    # 采样密度按像素弧长决定：弧长 1px 一个点，保证墨迹沉积不出现空隙
    raw_length = _arc_length(xy)
    sample_count = int(min(180.0, max(10.0, raw_length * em_px * 0.85)))
    path = _resample_uniform(xy, sample_count)

    # 圆转：转角圆化半径越大，平滑越强
    path = _smooth_path(path, brush.corner_radius * em_px * 0.55)

    # 外弧偏置：让笔画"鼓"起来，印刷骨架是几何直线，手写不是
    if brush.curvature_bias > 1e-4 and profile.bow != 0.0 and len(path) >= 5:
        fractions, _ = _arc_params(path)
        bow = np.sin(np.pi * fractions) * brush.curvature_bias * profile.bow
        path = path + _normals(path) * bow[:, None]

    fractions, total_length = _arc_params(path)
    curvature = _curvature(path)
    curvature_norm = curvature / (quantile(curvature, 0.9) + 1e-9)

    # 速度：对数正态包络 × 2/3 幂律（曲率越大越慢）× 笔型速度倍率
    envelope = _lognormal_speed(
        fractions, _submovements(fractions, curvature), motion.lognormal_mu, motion.lognormal_sigma
    )
    power_law = np.power(1.0 + 2.2 * np.clip(curvature_norm, 0.0, 6.0), -1.0 / 3.0)
    speed_noise = 1.0 + _ou_noise(len(path), motion.speed_cv * 0.45, 0.92, rng)
    speed = motion.mean_speed * profile.speed_scale * envelope * power_law * speed_noise
    speed = np.clip(speed, motion.mean_speed * 0.06, motion.mean_speed * 3.2)

    # 时间：t(s) = ∫ ds / v(s)
    segment_length = np.diff(fractions) * total_length
    midpoint_speed = 0.5 * (speed[:-1] + speed[1:])
    dt = segment_length / np.maximum(midpoint_speed, 1e-6)
    time_s = start_time + profile.entry_pause_s + np.concatenate([[0.0], np.cumsum(dt)])

    # 宽度倍率：笔型剖面（起笔顿、中段提、收笔顿/出锋）
    width_scale = evaluate_width_profile(type_id, fractions)
    # 藏锋/露锋：藏锋起笔更重，露锋起笔更尖
    entry_zone = np.clip(fractions / max(brush.entry_length, 1e-3), 0.0, 1.0)
    width_scale *= 1.0 - (1.0 - brush.entry_hidden) * 0.55 * (1.0 - entry_zone)
    # 顿笔/出锋
    exit_zone = np.clip((1.0 - fractions) / max(brush.exit_taper, 1e-3), 0.0, 1.0)
    if profile.tapers_out:
        taper = 0.06 + 0.94 * np.power(exit_zone, 0.65)
        if USE_LEGACY_INK:
            # 补丁前：第一项恒为 1，exit_pressed 在撇捺钩上不生效
            width_scale *= taper
        else:
            # 顿笔压住出锋：1=保持剖面宽度，0=走尖收
            width_scale *= brush.exit_pressed + (1.0 - brush.exit_pressed) * taper
    else:
        width_scale *= 1.0 + (brush.exit_pressed - 0.5) * 0.42 * (1.0 - exit_zone)

    # 压力：速度反相关 + 曲率正相关 + 提按低频调制 + 生理噪声
    speed_norm = np.clip(speed / (motion.mean_speed * profile.speed_scale + 1e-9), 0.0, 3.0)
    pressure = (
        0.62
        * (1.0 - 0.22 * np.clip(speed_norm - 1.0, -1.0, 2.0))
        * (1.0 + 0.30 * np.clip(curvature_norm, 0.0, 3.0))
    )
    tiba = 1.0 + brush.tiba_depth * np.sin(
        fractions * math.pi * (1.4 + rng.random() * 1.3) + rng.random() * math.tau
    )
    pressure = pressure * tiba * (1.0 + _ou_noise(len(path), 0.055, 0.90, rng))
    pressure = np.clip(pressure, 0.05, 1.0)

    # 生理微颤：沿笔画相关，幅度亚像素
    if motion.tremor > 1e-6 and len(path) >= 4:
        phase = rng.random() * math.tau
        tremor = motion.tremor * np.sin(fractions * math.tau * 5.7 + phase)
        path = path + _normals(path) * tremor[:, None]

    return PenPath(
        xy=path,
        time_s=time_s,
        pressure=pressure,
        width_scale=np.clip(width_scale, 0.03, 2.2),
        speed=speed,
        type_id=type_id,
        is_ligature=False,
    )


def _make_ligature(
    previous: PenPath, following: PenPath, style: WriterStyle, rng: np.random.Generator, em_px: float
) -> PenPath | None:
    motion = style.motion
    start = previous.xy[-1]
    end = following.xy[0]
    gap = float(np.hypot(*(end - start)))
    if gap > motion.ligature_distance or gap < 1e-3:
        return None
    if rng.random() > motion.ligature_propensity:
        return None

    # 切向由前一笔的收笔方向与后一笔的起笔方向决定 —— 牵丝是运动的延续
    exit_dir = previous.xy[-1] - previous.xy[max(0, len(previous.xy) - 4)]
    entry_dir = following.xy[min(3, len(following.xy) - 1)] - following.xy[0]
    exit_dir = exit_dir / (np.linalg.norm(exit_dir) + 1e-9)
    entry_dir = entry_dir / (np.linalg.norm(entry_dir) + 1e-9)
    handle = gap * 0.42
    control_a = start + exit_dir * handle
    control_b = end - entry_dir * handle

    count = int(min(90.0, max(8.0, gap * em_px * 1.2)))
    u = np.linspace(0.0, 1.0, count)[:, None]
    curve = (
        (1 - u) ** 3 * start
        + 3 * (1 - u) ** 2 * u * control_a
        + 3 * (1 - u) * u**2 * control_b
        + u**3 * end
    )

    fractions = np.linspace(0.0, 1.0, count)
    # 牵丝中段最细，两端与主笔衔接
    taper = np.sin(np.pi * fractions) ** 1.4
    width = motion.ligature_pressure * (0.45 + 0.55 * (1.0 - taper))
    pressure = np.clip(motion.ligature_pressure * (1.0 - 0.65 * taper), 0.02, 0.5)
    speed = np.full(count, motion.mean_speed * 1.9)
    duration = gap / max(speed[0], 1e-6)
    time_s = previous.time_s[-1] + fractions * duration

    return PenPath(
        xy=curve,
        time_s=time_s,
        pressure=pressure,
        width_scale=np.clip(width, 0.02, 0.5),
        speed=speed,
        type_id=previous.type_id,
        is_ligature=True,
    )


def synthesize(
    skeleton: GlyphSkeleton,
    style: WriterStyle,
    *,
    seed: int,
    occurrence: int,
    em_px: float,
) -> list[PenPath]:
    """合成一个字的一次书写。

    ``occurrence`` 让同一个字的每一次出现都不同；同样的
    (seed, char, occurrence, style) 必然得到完全相同的结果（开发原则 #5）。
    """
    rng = _rng(seed, style.style_id, skeleton.char, occurrence)

    strokes = [skeleton.points[index].astype(np.float64) for index in range(skeleton.stroke_count)]
    strokes = _apply_structure(strokes, skeleton, style, rng)
    strokes = _elastic_deform(strokes, style, rng)

    paths: list[PenPath] = []
    clock = 0.0
    for index, xy in enumerate(strokes):
        type_id = int(skeleton.type_ids[index])
        path = _build_stroke_path(xy, type_id, style, rng, clock, em_px)
        if paths:
            ligature = _make_ligature(paths[-1], path, style, rng, em_px)
            if ligature is not None:
                paths.append(ligature)
        paths.append(path)
        clock = float(path.time_s[-1]) + profile_for(type_id).exit_pause_s
        clock += style.motion.junction_pause_s * (0.6 + 0.8 * rng.random())

    return paths
