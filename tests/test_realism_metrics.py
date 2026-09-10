"""Unit tests for realism metrics helpers."""

import numpy as np

from renderer.metrics.realism import (
    ReferenceStats,
    emd_1d,
    ks_distance,
    measure_image,
    normalized_hist,
)


def test_ks_identical_is_zero():
    a = np.linspace(0, 1, 50)
    assert ks_distance(a, a) == 0.0


def test_emd_identical_is_zero():
    h = normalized_hist(np.random.default_rng(0).random(100), bins=32)
    assert emd_1d(h, h) == 0.0


def test_ks_different_is_positive():
    a = np.zeros(20)
    b = np.ones(20)
    assert ks_distance(a, b) > 0.5


def _page(seed: int) -> np.ndarray:
    """一张假的"手写页"：横向墨条 + 纸面噪点。够真实到让每个指标都有值。"""
    rng = np.random.default_rng(seed)
    gray = np.full((320, 240), 0.93, dtype=np.float32)
    gray += rng.normal(0.0, 0.01, gray.shape).astype(np.float32)
    for row in range(20, 300, 18):
        for start in range(15, 220, 12):
            width = 2 + (seed + row + start) % 3
            gray[row : row + width, start : start + 8] = 0.18
    return np.clip(gray, 0.0, 1.0)


def test_reference_digest_gives_the_same_numbers_as_the_originals():
    """摘要必须能顶替原图。

    真人照片是敏感数据不入库，CI 上只有这份摘要；两条路算出来的指标不一致的话，
    本地绿、CI 红（或者更糟：CI 绿但量的是别的东西）。
    """
    refs = ReferenceStats.from_grays([_page(s) for s in (1, 2, 3, 4)])
    revived = ReferenceStats.from_dict(refs.to_dict())
    assert revived.images == refs.images

    probe = _page(9)
    direct = measure_image(probe, refs)
    from_digest = measure_image(probe, revived)
    for name, value in direct.items():
        assert from_digest[name] == value, name


def test_reference_digest_carries_no_pixels():
    """摘要里只能有聚合量。有人往里加原图字段的话，这条会拦住。"""
    payload = ReferenceStats.from_grays([_page(s) for s in (1, 2, 3)]).to_dict()
    assert set(payload) == {"images", "width_counts", "density", "slopes", "psd_profiles"}
    assert len(payload["psd_profiles"][0]) <= 64
