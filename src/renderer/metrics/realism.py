"""Automated realism metrics (numpy + PIL only).

Compare synthetic handwriting renders against a reference corpus of real scans.
Used by ``scripts/realism_gate.py``; thresholds come from corpus calibration, not a fixed plan doc.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    threshold: float
    comparator: str  # "le" | "ge" | "band"
    passed: bool
    detail: str = ""
    lower: float | None = None


def _passes(value: float, threshold: float, comparator: str, lower: float | None) -> bool:
    if comparator == "band":
        return (lower is None or value >= lower) and value <= threshold
    if comparator == "ge":
        return value >= threshold
    return value <= threshold


# 兜底阈值：没有校准文件时使用。真实门禁应当走 calibrate_references 产出的
# 真人分布带 —— 这几个数字来自方案文档，未经参照语料验证。
FALLBACK_THRESHOLDS: dict[str, tuple[float, str]] = {
    "stroke_width_ks": (0.12, "le"),
    "density_hist_emd": (0.08, "le"),
    "baseline_psd_slope": (0.25, "le"),
    "glyph_selfvar": (0.002, "ge"),
    "ink_paper_noise_match": (0.10, "band"),
    "radial_psd_match": (0.92, "ge"),
}

# le / ge 适用于"到真人语料的距离"，越接近越好，比真人还接近不算问题。
# band 适用于"图像自身的统计量"：真人照片本身就落在一个区间里，高于上界和
# 低于下界一样不真实。
_BAND_METRICS = frozenset({"ink_paper_noise_match"})

# glyph_selfvar 测的是"同文本换 seed 重写"的差异，真人语料里没有对应量，
# 因此它不参与分布带校准，保留为引擎内部下限。
UNCALIBRATED_METRICS = frozenset({"glyph_selfvar"})


def load_gray(path: str | Path, max_side: int = 1600) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return np.asarray(ImageOps.grayscale(img), dtype=np.float32) / 255.0


def ink_mask(gray: np.ndarray, threshold: float = 0.62) -> np.ndarray:
    return gray < threshold


def _laplacian(gray: np.ndarray) -> np.ndarray:
    up = np.roll(gray, -1, axis=0)
    down = np.roll(gray, 1, axis=0)
    left = np.roll(gray, -1, axis=1)
    right = np.roll(gray, 1, axis=1)
    return up + down + left + right - 4.0 * gray


def gray_from_image(img: Image.Image, max_side: int = 1600) -> np.ndarray:
    rgb = img.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        rgb = rgb.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return np.asarray(ImageOps.grayscale(rgb), dtype=np.float32) / 255.0


# 一笔的宽度上限，占页高比例。手写笔画最宽约 6mm，A4 页高 297mm -> 2%。
# 超过这个宽度的暗色横条不是笔画，而是拍照背景、桌面阴影或纸张暗边：真人
# 照片里这类横条能长到整幅图宽（实测 1440px），不剔除会淹没真正的笔画分布。
MAX_STROKE_WIDTH_RATIO = 0.02


def stroke_width_samples(
    gray: np.ndarray,
    mask: np.ndarray,
    limit: int = 4000,
    max_width_ratio: float = MAX_STROKE_WIDTH_RATIO,
) -> np.ndarray:
    """Stroke widths from horizontal ink runs, one sample per run."""
    height, width = mask.shape
    max_run = max(3, int(round(height * max_width_ratio)))
    padded = np.zeros((height, width + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    delta = np.diff(padded, axis=1)
    # 每行首尾补零，行内 start/end 必然成对且顺序一致
    starts = np.argwhere(delta == 1)
    ends = np.argwhere(delta == -1)
    if len(starts) == 0 or len(starts) != len(ends):
        return np.array([0.0], dtype=np.float32)
    runs = (ends[:, 1] - starts[:, 1]).astype(np.float32)
    runs = runs[(runs >= 1.0) & (runs <= max_run)]
    if runs.size == 0:
        return np.array([0.0], dtype=np.float32)
    if runs.size > limit:
        idx = np.random.default_rng(0).choice(runs.size, size=limit, replace=False)
        runs = runs[idx]
    return runs


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a.astype(np.float64))
    b = np.sort(b.astype(np.float64))
    if len(a) == 0 or len(b) == 0:
        return 1.0
    data = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, data, side="right") / len(a)
    cdf_b = np.searchsorted(b, data, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def normalized_hist(values: np.ndarray, bins: int = 64) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=True)
    hist = hist.astype(np.float64)
    s = hist.sum()
    return hist / s if s > 0 else hist


def emd_1d(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p /= p.sum() if p.sum() > 0 else 1.0
    q /= q.sum() if q.sum() > 0 else 1.0
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q)))) / len(p)


def density_hist(gray: np.ndarray, mask: np.ndarray, bins: int = 64) -> np.ndarray:
    ink = gray[mask]
    if ink.size == 0:
        return np.zeros(bins, dtype=np.float64)
    # optical density proxy in [0, 1]
    density = 1.0 - ink
    return normalized_hist(density, bins=bins)


def _box_mean3(gray: np.ndarray) -> np.ndarray:
    pad = np.pad(gray, 1, mode="edge")
    total = (
        pad[0:-2, 0:-2] + pad[0:-2, 1:-1] + pad[0:-2, 2:]
        + pad[1:-1, 0:-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:]
        + pad[2:, 0:-2] + pad[2:, 1:-1] + pad[2:, 2:]
    )
    return total / 9.0


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask
    for _ in range(max(0, radius)):
        out = (
            out
            & np.roll(out, 1, 0)
            & np.roll(out, -1, 0)
            & np.roll(out, 1, 1)
            & np.roll(out, -1, 1)
        )
    return out


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    return ~_erode(~mask, radius)


def _residual_scale(values: np.ndarray) -> float | None:
    """High-pass energy of a region.

    Trimmed at the 98th percentile so a handful of surviving edge pixels cannot
    set the scale, but not a median deviation -- real scans JPEG-flatten smooth
    paper to literal constants, which would make a median-based scale zero.
    """
    if values.size < 64:
        return None
    magnitude = np.abs(values - np.median(values))
    cutoff = np.quantile(magnitude, 0.98)
    trimmed = magnitude[magnitude <= cutoff] if cutoff > 0 else magnitude
    return float(np.sqrt(np.mean(np.square(trimmed))))


def ink_paper_noise_delta(gray: np.ndarray, mask: np.ndarray) -> float:
    """Do ink and paper carry the same high-frequency noise?

    Synthetic pages give themselves away by drawing clean ink onto a quiet page:
    the two regions end up with different grain. Measured on the high-pass
    residual, over **stroke interiors and paper well away from strokes** -- the
    transition band around every stroke is excluded because its high-frequency
    energy comes from the stroke edge, which would turn this into a measure of
    ink coverage instead of noise consistency.
    """
    residual = gray - _box_mean3(gray)
    ink_core = _erode(mask, 1)
    paper_core = ~_dilate(mask, 3)
    ink_scale = _residual_scale(residual[ink_core])
    paper_scale = _residual_scale(residual[paper_core])
    if ink_scale is None:
        ink_scale = _residual_scale(residual[mask])
    if ink_scale is None or paper_scale is None:
        return 1.0
    return float(abs(ink_scale - paper_scale) / (ink_scale + paper_scale + 1e-6))


def radial_psd_profile(gray: np.ndarray, bins: int = 48) -> np.ndarray:
    f = np.fft.fft2(gray - gray.mean())
    power = np.abs(np.fft.fftshift(f)) ** 2
    cy, cx = np.array(power.shape) // 2
    ys, xs = np.indices(power.shape)
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    r_int = r.astype(np.int32).ravel()
    p = power.ravel()
    max_r = int(r.max())
    prof = np.zeros(max_r + 1, dtype=np.float64)
    counts = np.zeros(max_r + 1, dtype=np.float64)
    for ri, val in zip(r_int, p):
        prof[ri] += val
        counts[ri] += 1
    counts[counts == 0] = 1.0
    prof /= counts
    if len(prof) > bins:
        step = len(prof) // bins
        prof = prof[: step * bins].reshape(bins, step).mean(axis=1)
    s = np.linalg.norm(prof)
    return prof / s if s > 0 else prof


def radial_psd_match(gray: np.ndarray, ref_profiles: Sequence[np.ndarray]) -> float:
    """与每张参照图的径向功率谱做相关，取均值。

    参数收的是**谱线**而不是图。这是参照侧唯一还需要整张图的指标，改成收谱线
    之后，门禁就能只靠一份摘要跑起来，不必带着真人照片走 —— 见 ``ReferenceStats``。
    """
    prof = radial_psd_profile(gray)
    corrs = []
    for rprof in ref_profiles:
        n = min(len(prof), len(rprof))
        a, b = prof[:n], rprof[:n]
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            continue
        corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(corrs)) if corrs else 0.0


def baseline_psd_slope(gray: np.ndarray) -> float:
    row_ink = (gray < 0.62).mean(axis=1)
    if row_ink.std() < 1e-6:
        return 0.0
    y = row_ink - row_ink.mean()
    f = np.fft.rfft(y)
    power = np.abs(f) ** 2
    freqs = np.arange(len(power)) + 1
    mask = (freqs > 1) & (power > 1e-9)
    if mask.sum() < 3:
        return 0.0
    log_f = np.log(freqs[mask].astype(np.float64))
    log_p = np.log(power[mask].astype(np.float64))
    slope = np.polyfit(log_f, log_p, 1)[0]
    return float(slope)


def glyph_selfvar(images: Sequence[np.ndarray]) -> float:
    """Mean pixel variance across repeated renders (higher = more natural variation)."""
    if len(images) < 2:
        return 0.0
    stack = np.stack(images, axis=0).astype(np.float32)
    return float(stack.std(axis=0).mean())


@dataclass(frozen=True)
class ReferenceStats:
    """参照语料的摘要。合成页的每个指标都只跟这份摘要比。

    存在的理由是**门禁不该依赖原图**：真人手写照片属于敏感数据，按仓库策略不
    入库（见 .gitignore），于是 CI 里那个"真实度回归"步骤永远跑不起来 —— 一个
    永远红的门禁等于没有门禁。

    这里的四项都是不可逆的聚合量：笔宽直方图、墨浓直方图、行频谱斜率、径向
    功率谱曲线（48 个归一化数）。从它们复原不出任何一张照片，也认不出是谁写的，
    所以可以随仓库分发。有原图时照常从原图重建，没有时用摘要，结果一致。
    """

    psd_profiles: list[np.ndarray]
    widths: np.ndarray
    density: np.ndarray
    slopes: list[float]
    images: int = 0

    @classmethod
    def from_paths(cls, ref_paths: Iterable[str | Path]) -> "ReferenceStats":
        grays = [load_gray(path) for path in ref_paths]
        return cls.from_grays(grays)

    @classmethod
    def from_grays(cls, grays: Sequence[np.ndarray]) -> "ReferenceStats":
        widths: list[np.ndarray] = []
        densities: list[np.ndarray] = []
        slopes: list[float] = []
        profiles: list[np.ndarray] = []
        for gray in grays:
            mask = ink_mask(gray)
            widths.append(stroke_width_samples(gray, mask, limit=1500))
            densities.append(density_hist(gray, mask))
            slopes.append(baseline_psd_slope(gray))
            profiles.append(radial_psd_profile(gray))
        return cls(
            psd_profiles=profiles,
            widths=np.concatenate(widths) if widths else np.array([1.0], dtype=np.float32),
            density=(
                np.mean(np.stack(densities, axis=0), axis=0)
                if densities
                else np.zeros(64, dtype=np.float64)
            ),
            slopes=slopes,
            images=len(grays),
        )

    def to_dict(self) -> dict:
        # 笔宽本身是整数行程，存计数表而不是几万个样本；KS 距离只看经验分布，
        # 计数表能一比一还原它。
        values, counts = np.unique(self.widths.astype(np.int64), return_counts=True)
        return {
            "images": self.images,
            "width_counts": {str(int(v)): int(c) for v, c in zip(values, counts)},
            "density": [float(x) for x in self.density],
            "slopes": [float(x) for x in self.slopes],
            "psd_profiles": [[float(x) for x in prof] for prof in self.psd_profiles],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ReferenceStats":
        counts = payload.get("width_counts") or {}
        widths = np.repeat(
            np.array([float(k) for k in counts], dtype=np.float32),
            np.array([int(v) for v in counts.values()], dtype=np.int64),
        ) if counts else np.array([1.0], dtype=np.float32)
        return cls(
            psd_profiles=[
                np.asarray(prof, dtype=np.float64) for prof in payload.get("psd_profiles", [])
            ],
            widths=widths,
            density=np.asarray(payload.get("density") or [], dtype=np.float64),
            slopes=[float(x) for x in payload.get("slopes") or []],
            images=int(payload.get("images") or 0),
        )


def measure_image(
    gray: np.ndarray,
    refs: ReferenceStats,
    *,
    selfvar: float | None = None,
) -> dict[str, float]:
    """Raw metric values for one image against the reference corpus."""
    mask = ink_mask(gray)
    slope = baseline_psd_slope(gray)
    return {
        "stroke_width_ks": ks_distance(stroke_width_samples(gray, mask), refs.widths),
        "density_hist_emd": emd_1d(density_hist(gray, mask), refs.density),
        "baseline_psd_slope": (
            float(np.mean([abs(slope - rs) for rs in refs.slopes])) if refs.slopes else 0.0
        ),
        "glyph_selfvar": float(selfvar) if selfvar is not None else 0.0,
        "ink_paper_noise_match": ink_paper_noise_delta(gray, mask),
        "radial_psd_match": radial_psd_match(gray, refs.psd_profiles),
    }


def evaluate_image(
    gray: np.ndarray,
    refs: ReferenceStats,
    *,
    selfvar: float | None = None,
    thresholds: dict[str, tuple[float, str]] | None = None,
) -> list[MetricResult]:
    values = measure_image(gray, refs, selfvar=selfvar)
    limits = thresholds or FALLBACK_THRESHOLDS
    out: list[MetricResult] = []
    for name, value in values.items():
        spec = limits.get(name, FALLBACK_THRESHOLDS[name])
        threshold, comp = spec[0], spec[1]
        lower = spec[2] if len(spec) > 2 else None
        out.append(
            MetricResult(name, value, threshold, comp, _passes(value, threshold, comp, lower), lower=lower)
        )
    return out


def calibrate_references(ref_paths: Iterable[str | Path]) -> dict[str, dict[str, float]]:
    """Score every real photo against the other real photos (leave-one-out).

    This answers the only question that makes an absolute threshold meaningful:
    *what does a genuine handwriting photo score?* A synthetic page is realistic
    when it lands inside this band, not when it beats a number someone wrote in a
    design document.
    """
    paths = list(ref_paths)
    if len(paths) < 3:
        raise ValueError("calibration needs at least 3 reference images")
    grays = [load_gray(p) for p in paths]
    masks = [ink_mask(g) for g in grays]
    widths = [stroke_width_samples(g, m, limit=1500) for g, m in zip(grays, masks)]
    densities = [density_hist(g, m) for g, m in zip(grays, masks)]
    slopes = [baseline_psd_slope(g) for g in grays]
    profiles = [radial_psd_profile(g) for g in grays]

    scores: dict[str, list[float]] = {
        name: [] for name in FALLBACK_THRESHOLDS if name not in UNCALIBRATED_METRICS
    }
    for i, gray in enumerate(grays):
        others = [j for j in range(len(grays)) if j != i]
        held_out = measure_image(
            gray,
            ReferenceStats(
                psd_profiles=[profiles[j] for j in others],
                widths=np.concatenate([widths[j] for j in others]),
                density=np.mean(np.stack([densities[j] for j in others]), axis=0),
                slopes=[slopes[j] for j in others],
                images=len(others),
            ),
        )
        for name in scores:
            scores[name].append(held_out[name])

    bands: dict[str, dict[str, float]] = {}
    for name, values in scores.items():
        arr = np.asarray(values, dtype=np.float64)
        bands[name] = {
            "p10": float(np.percentile(arr, 10)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "comparator": FALLBACK_THRESHOLDS[name][1],
        }
    return bands


def thresholds_from_bands(bands: dict[str, dict[str, float]]) -> dict[str, tuple]:
    """Human band -> pass/fail limit."""
    limits: dict[str, tuple] = dict(FALLBACK_THRESHOLDS)
    for name, band in bands.items():
        comp = str(band.get("comparator") or FALLBACK_THRESHOLDS[name][1])
        if comp == "band":
            limits[name] = (float(band["p90"]), comp, float(band["p10"]))
        else:
            limits[name] = (float(band["p90"] if comp == "le" else band["p10"]), comp)
    return limits


def build_reference_stats(ref_paths: Iterable[str | Path]) -> ReferenceStats:
    return ReferenceStats.from_paths(ref_paths)
