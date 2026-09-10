"""笔型常量与宽度剖面 —— 渲染层与构建工具共用的唯一定义（开发原则 #1）。

笔型 id 是稳定契约：骨架库二进制里存的就是这些整数，改动等于让已构建的库失效。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

HENG: Final = 0
SHU_CHUILU: Final = 1
SHU_XUANZHEN: Final = 2
PIE: Final = 3
NA: Final = 4
DIAN: Final = 5
TI: Final = 6
ZHE: Final = 7
GOU: Final = 8
ZHE_GOU: Final = 9
WAN: Final = 10
UNKNOWN: Final = 11

STROKE_TYPE_NAMES: Final[dict[int, str]] = {
    HENG: "heng",
    SHU_CHUILU: "shu-chuilu",
    SHU_XUANZHEN: "shu-xuanzhen",
    PIE: "pie",
    NA: "na",
    DIAN: "dian",
    TI: "ti",
    ZHE: "zhe",
    GOU: "gou",
    ZHE_GOU: "zhe-gou",
    WAN: "wan",
    UNKNOWN: "unknown",
}


@dataclass(frozen=True)
class StrokeProfile:
    """一个笔型的物理特征。"""

    type_id: int
    # 相对标称笔宽的 5 个控制点，覆盖弧长 0 → 1。这是"提按"的形状来源。
    width_profile: tuple[float, float, float, float, float]
    ink_weight: float
    entry_pause_s: float
    exit_pause_s: float
    speed_scale: float
    tapers_out: bool
    # 笔画外弧方向与强度（相对 curvature_bias 的倍率，正=向左法线侧鼓）
    bow: float


_PROFILES: Final[dict[int, StrokeProfile]] = {
    HENG: StrokeProfile(HENG, (0.92, 0.74, 0.70, 0.80, 1.06), 1.00, 0.028, 0.032, 1.12, False, 0.55),
    SHU_CHUILU: StrokeProfile(SHU_CHUILU, (0.95, 0.90, 0.88, 0.94, 1.02), 1.05, 0.030, 0.034, 1.00, False, -0.28),
    SHU_XUANZHEN: StrokeProfile(SHU_XUANZHEN, (0.96, 0.92, 0.76, 0.42, 0.07), 1.00, 0.030, 0.004, 1.06, True, -0.20),
    PIE: StrokeProfile(PIE, (1.04, 0.88, 0.62, 0.33, 0.06), 0.98, 0.026, 0.004, 1.18, True, 0.95),
    NA: StrokeProfile(NA, (0.26, 0.48, 0.76, 1.32, 0.34), 1.22, 0.010, 0.040, 0.82, True, -0.70),
    DIAN: StrokeProfile(DIAN, (0.38, 0.84, 1.18, 1.02, 0.52), 1.10, 0.012, 0.018, 0.90, False, 0.30),
    TI: StrokeProfile(TI, (1.08, 0.82, 0.52, 0.26, 0.05), 0.94, 0.024, 0.003, 1.24, True, 0.25),
    ZHE: StrokeProfile(ZHE, (0.94, 0.80, 0.86, 0.82, 0.98), 1.04, 0.028, 0.030, 0.88, False, 0.18),
    GOU: StrokeProfile(GOU, (0.94, 0.88, 0.84, 0.60, 0.08), 1.02, 0.028, 0.004, 0.96, True, -0.22),
    ZHE_GOU: StrokeProfile(ZHE_GOU, (0.94, 0.82, 0.86, 0.58, 0.08), 1.06, 0.028, 0.004, 0.84, True, 0.14),
    WAN: StrokeProfile(WAN, (0.90, 0.86, 0.88, 0.84, 0.72), 1.02, 0.026, 0.020, 0.90, False, 0.40),
    UNKNOWN: StrokeProfile(UNKNOWN, (0.92, 0.88, 0.88, 0.88, 0.92), 1.00, 0.026, 0.024, 1.00, False, 0.20),
}


def profile_for(type_id: int) -> StrokeProfile:
    return _PROFILES.get(int(type_id), _PROFILES[UNKNOWN])


def all_profiles() -> dict[int, StrokeProfile]:
    return dict(_PROFILES)


def evaluate_width_profile(type_id: int, s: np.ndarray) -> np.ndarray:
    """在归一化弧长 s∈[0,1] 上求宽度剖面，Catmull-Rom 插值 5 个控制点。"""
    control = np.asarray(profile_for(type_id).width_profile, dtype=np.float64)
    # 端点各复制一份，让 Catmull-Rom 在边界有完整的四点邻域
    padded = np.concatenate([control[:1], control, control[-1:]])
    span = len(control) - 1
    position = np.clip(s, 0.0, 1.0) * span
    index = np.clip(position.astype(np.int64), 0, span - 1)
    u = (position - index)[:, None]
    p0 = padded[index]
    p1 = padded[index + 1]
    p2 = padded[index + 2]
    p3 = padded[index + 3]
    u2 = u[:, 0] ** 2
    u3 = u2 * u[:, 0]
    return 0.5 * (
        2 * p1
        + (-p0 + p2) * u[:, 0]
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * u3
    )
