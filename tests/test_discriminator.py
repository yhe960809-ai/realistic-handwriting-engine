"""真伪判别器：分不开时接近 0.5，明显不同时接近 1.0，且不因分组泄漏虚高。"""

import numpy as np

from renderer.metrics.discriminator import PATCH, evaluate, patch_features


def _page(seed: int, *, noise: float, stroke_width: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    page = np.ones((PATCH * 3, PATCH * 3), dtype=np.float32) * 0.96
    for y in range(20, page.shape[0] - 20, 34):
        for x in range(20, page.shape[1] - 40, 46):
            page[y : y + stroke_width, x : x + 30] = 0.15
            page[y : y + 26, x : x + stroke_width] = 0.18
    page += rng.normal(0.0, noise, page.shape).astype(np.float32)
    return np.clip(page, 0.0, 1.0)


def test_features_are_finite():
    features = patch_features(_page(1, noise=0.02)[:PATCH, :PATCH])
    assert features.shape == (8,)
    assert np.isfinite(features).all()


def test_identical_distributions_are_not_separable():
    real = [_page(s, noise=0.02) for s in range(4)]
    fake = [_page(s, noise=0.02) for s in range(100, 104)]
    report = evaluate(real, fake)
    assert report.folds >= 4
    # 同一分布的两组，判别器不该分得开；留一交叉验证下允许小幅偏离 0.5
    assert 0.2 <= report.auc <= 0.8


def test_obviously_cleaner_pages_are_separable():
    real = [_page(s, noise=0.06) for s in range(4)]
    fake = [_page(s, noise=0.0005) for s in range(100, 104)]
    report = evaluate(real, fake)
    assert report.auc > 0.9
    assert report.per_feature["residual_rms"]["auc"] > 0.8


def test_report_serialises():
    payload = evaluate([_page(1, noise=0.03)], [_page(2, noise=0.001)]).as_dict()
    assert set(payload) >= {"auc", "folds", "real_patches", "fake_patches", "per_feature"}
    assert "residual_rms" in payload["per_feature"]
