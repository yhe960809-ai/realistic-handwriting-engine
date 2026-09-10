"""真伪判别器：一个简单分类器能不能把合成页和真人照片分开。

方案文档写的是训练 CNN（``discriminator_auc ≤ 0.65``）。参照语料只有 13 张真人
照片，CNN 在这个量级上学到的是"哪张照片"而不是"真还是假"，AUC 会漂亮得毫无
意义。这里改用**低容量线性分类器 + 按图留一交叉验证**：

- 特征是取证意义明确的手工统计量（高频残差、笔画宽度、墨纸噪声比、JPEG 块效
  应、梯度方向熵），每一维都能解释成"哪里露馅了"；
- 同一张图切出的所有 patch 只能整体出现在训练集或测试集，否则模型认图不认真伪；
- AUC 0.5 = 分不开（理想），1.0 = 一眼假。

它衡量的是"这些统计量上分不分得开"，不是"人眼分不分得开"。人眼盲测不可替代。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .realism import (
    _box_mean3,
    _dilate,
    _erode,
    _residual_scale,
    ink_mask,
    load_gray,
    stroke_width_samples,
)

PATCH = 256
FEATURE_NAMES = (
    "residual_rms",
    "ink_ratio",
    "stroke_median",
    "stroke_iqr",
    "ink_paper_residual_ratio",
    "jpeg_blockiness",
    "orientation_entropy",
    "spectral_slope",
)


@dataclass(frozen=True)
class DiscriminatorReport:
    auc: float
    folds: int
    real_patches: int
    fake_patches: int
    # 每个特征单独的判别力与两侧中位数。总 AUC 只说"分得开"，这张表说"哪里露馅"。
    per_feature: dict

    def as_dict(self) -> dict:
        return {
            "auc": round(self.auc, 4),
            "folds": self.folds,
            "real_patches": self.real_patches,
            "fake_patches": self.fake_patches,
            "per_feature": self.per_feature,
        }


def _patches(gray: np.ndarray, limit: int = 6) -> list[np.ndarray]:
    """取墨迹最多的若干块。空白纸块两边都一样，对判别毫无信息量。"""
    height, width = gray.shape
    if height < PATCH or width < PATCH:
        return []
    ys = range(0, height - PATCH + 1, PATCH)
    xs = range(0, width - PATCH + 1, PATCH)
    scored: list[tuple[float, np.ndarray]] = []
    for y in ys:
        for x in xs:
            tile = gray[y : y + PATCH, x : x + PATCH]
            coverage = float((tile < 0.62).mean())
            if coverage < 0.02:
                continue
            scored.append((coverage, tile))
    scored.sort(key=lambda item: -item[0])
    return [tile for _, tile in scored[:limit]]


def _spectral_slope(tile: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft2(tile - tile.mean())) ** 2
    rows = np.fft.fftfreq(tile.shape[0])[:, None]
    cols = np.fft.rfftfreq(tile.shape[1])[None, :]
    radius = np.sqrt(rows**2 + cols**2)
    mask = radius > 0.02
    if mask.sum() < 16:
        return 0.0
    log_r = np.log(radius[mask])
    log_p = np.log(spectrum[mask] + 1e-12)
    return float(np.polyfit(log_r, log_p, 1)[0])


def _blockiness(tile: np.ndarray) -> float:
    """8 像素网格上的阶跃能量相对其余位置。JPEG 块效应的直接度量。"""
    diff = np.abs(np.diff(tile, axis=1))
    if diff.shape[1] < 16:
        return 0.0
    columns = np.arange(diff.shape[1])
    on_grid = diff[:, columns % 8 == 7]
    off_grid = diff[:, columns % 8 != 7]
    if on_grid.size == 0 or off_grid.size == 0:
        return 0.0
    return float(on_grid.mean() / (off_grid.mean() + 1e-9))


def _orientation_entropy(tile: np.ndarray) -> float:
    gy, gx = np.gradient(tile)
    magnitude = np.hypot(gx, gy)
    strong = magnitude > np.quantile(magnitude, 0.85)
    if strong.sum() < 64:
        return 0.0
    angles = np.arctan2(gy[strong], gx[strong]) % np.pi
    hist, _ = np.histogram(angles, bins=18, range=(0.0, np.pi))
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0
    p /= total
    nonzero = p[p > 0]
    return float(-(nonzero * np.log(nonzero)).sum())


def patch_features(tile: np.ndarray) -> np.ndarray:
    mask = ink_mask(tile)
    residual = tile - _box_mean3(tile)
    widths = stroke_width_samples(tile, mask, limit=800)
    ink_scale = _residual_scale(residual[_erode(mask, 1)])
    paper_scale = _residual_scale(residual[~_dilate(mask, 3)])
    ratio = (
        float(ink_scale / (paper_scale + 1e-6))
        if ink_scale is not None and paper_scale is not None
        else 0.0
    )
    return np.array(
        [
            float(np.sqrt(np.mean(residual**2))),
            float(mask.mean()),
            float(np.median(widths)),
            float(np.percentile(widths, 75) - np.percentile(widths, 25)),
            min(ratio, 500.0),
            _blockiness(tile),
            _orientation_entropy(tile),
            _spectral_slope(tile),
        ],
        dtype=np.float64,
    )


def image_features(gray: np.ndarray, limit: int = 6) -> list[np.ndarray]:
    return [patch_features(tile) for tile in _patches(gray, limit=limit)]


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, steps: int = 400) -> np.ndarray:
    """带 L2 的批量梯度下降。低容量是有意为之：模型越强，测的越是它自己。

    样本按类别数倒数加权。留一交叉验证每次抽走一整张图，两类的样本数因此
    此消彼长；不做平衡的话截距会跟着类别比例漂，held-out 的那一类被系统性
    地推向另一侧 —— 两组分布完全相同时能测出 AUC=0.0 这种荒谬结果。
    """
    design = np.hstack([x, np.ones((x.shape[0], 1))])
    weights = np.zeros(design.shape[1])
    sample_weight = np.empty(len(y))
    for label in (0.0, 1.0):
        selector = y == label
        count = int(selector.sum())
        sample_weight[selector] = 0.0 if count == 0 else 0.5 / count
    learning_rate = 0.35
    l2 = 1.0
    for _ in range(steps):
        logits = design @ weights
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        gradient = design.T @ ((probability - y) * sample_weight)
        gradient[:-1] += l2 * weights[:-1] / len(y)
        weights -= learning_rate * gradient
    return weights


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return 0.5
    order = np.argsort(np.concatenate([negatives, positives]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    positive_ranks = ranks[negatives.size :]
    stat = positive_ranks.sum() - positives.size * (positives.size + 1) / 2.0
    return float(stat / (positives.size * negatives.size))


def evaluate(
    real_images: Sequence[np.ndarray], fake_images: Sequence[np.ndarray]
) -> DiscriminatorReport:
    """按图留一交叉验证的 AUC。同一张图的 patch 不跨越训练/测试。"""
    groups: list[np.ndarray] = []
    features: list[np.ndarray] = []
    labels: list[float] = []
    group_id = 0
    for label, images in ((0.0, real_images), (1.0, fake_images)):
        for gray in images:
            rows = image_features(gray)
            if not rows:
                continue
            for row in rows:
                features.append(row)
                labels.append(label)
                groups.append(group_id)
            group_id += 1
    if not features:
        return DiscriminatorReport(0.5, 0, 0, 0, {})

    x = np.vstack(features)
    y = np.asarray(labels)
    g = np.asarray(groups)
    if len(np.unique(y)) < 2:
        return DiscriminatorReport(0.5, 0, int((y == 0).sum()), int((y == 1).sum()), {})

    per_feature = {
        name: {
            "auc": round(max(_auc(x[:, i], y), 1.0 - _auc(x[:, i], y)), 3),
            "real_median": round(float(np.median(x[y == 0, i])), 4),
            "synthetic_median": round(float(np.median(x[y == 1, i])), 4),
        }
        for i, name in enumerate(FEATURE_NAMES)
    }

    scores = np.zeros(len(y))
    folds = 0
    for held_out in np.unique(g):
        train = g != held_out
        test = ~train
        if len(np.unique(y[train])) < 2:
            continue
        mean = x[train].mean(axis=0)
        scale = x[train].std(axis=0) + 1e-9
        weights = _fit_logistic((x[train] - mean) / scale, y[train])
        design = np.hstack([(x[test] - mean) / scale, np.ones((int(test.sum()), 1))])
        scores[test] = design @ weights
        folds += 1

    return DiscriminatorReport(
        auc=_auc(scores, y),
        folds=folds,
        real_patches=int((y == 0).sum()),
        fake_patches=int((y == 1).sum()),
        per_feature=per_feature,
    )


def evaluate_paths(
    real_paths: Iterable[str | Path], fake_grays: Sequence[np.ndarray]
) -> DiscriminatorReport:
    return evaluate([load_gray(p) for p in real_paths], fake_grays)
