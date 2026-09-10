"""书写者风格模型（Writer Style Model, WSM）。

一个人的"手"由四组参数刻画：结字、笔法、运笔、状态漂移。
同一份 WSM 配任意文本都应该稳定地像同一个人写的；
不同 WSM 之间的差异必须大于同一 WSM 内部的实例间差异 —— 这是风格可辨识性的定义。

所有参数都对应可解释的物理量或书法学概念，不使用无量纲魔法系数（开发原则 #3）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from . import legacy as _legacy

# 素材标定前的日常行楷默认值。HANDWRITE_V3_LEGACY_STYLE=1 时 get_style 还原这三项。
_LEGACY_XINGKAI_SLANT = 0.055
_LEGACY_XINGKAI_ASPECT = 1.0
_LEGACY_XINGKAI_SIZE_SIGMA = 0.042


@dataclass(frozen=True)
class Structure:
    """结字：字怎么摆。"""

    # 整体倾斜（弧度）。真人常见 -0.05 ~ +0.21（右倾为正）
    # 2026-08-28 由 E:\照片素材\素材 133 张干净整页中位数标定（原默认 +0.055）
    slant: float = 0.003
    # 横画上扬角（弧度）。中文手写的强特征，印刷体为 0
    heng_rise: float = 0.062
    # 重心偏移，em 比例
    gravity_dx: float = 0.0
    gravity_dy: float = -0.012
    # 中宫松紧：<1 收紧（字显瘦长有神），>1 放松
    zhonggong: float = 0.965
    # 字面宽高比修正
    # 2026-08-28 同上 133 张中位数 1.115（原默认 1.00）
    aspect: float = 1.12
    # 左右/上下结构的部件比例偏置：正值放大第一个部件
    component_ratio: float = -0.035
    # 部件相对位移抖动幅度（em 比例）
    component_jitter: float = 0.016
    # 单实例弹性形变强度（MLS 控制点扰动幅度，em 比例）
    elastic: float = 0.022


@dataclass(frozen=True)
class Brushwork:
    """笔法：每一笔怎么起、怎么收、怎么转。"""

    # 起笔藏锋比例（0=全露锋尖起，1=全藏锋顿起）
    entry_hidden: float = 0.55
    # 藏锋引入段长度（em 比例）
    entry_length: float = 0.022
    # 收笔顿笔比例（与出锋互补）
    exit_pressed: float = 0.5
    # 出锋长度（em 比例）
    exit_taper: float = 0.030
    # 转角圆化半径（em 比例）。写得快的人转角更圆
    corner_radius: float = 0.026
    # 钩的长度增益
    hook_gain: float = 1.0
    # 提按深度：压力调制幅度，0 = 匀速匀力（像字体），0.5 = 明显提按
    tiba_depth: float = 0.34
    # 笔画外弧偏置：正值让笔画更"鼓"
    curvature_bias: float = 0.030


@dataclass(frozen=True)
class Motion:
    """运笔：手的动力学。"""

    # 平均笔速（em/s）。真人日常书写约 3~9 em/s
    mean_speed: float = 5.2
    # 速度变异系数
    speed_cv: float = 0.26
    # Σ-Lognormal 形状参数：控制速度脉冲的偏斜与展宽
    lognormal_sigma: float = 0.34
    lognormal_mu: float = -1.15
    # 笔画间抬笔停顿（秒）
    junction_pause_s: float = 0.048
    # 连笔倾向：0 = 楷（笔笔断开），1 = 行草（能连就连）
    ligature_propensity: float = 0.22
    # 触发牵丝的笔间距上限（em 比例）
    ligature_distance: float = 0.34
    # 牵丝相对主笔的压力比
    ligature_pressure: float = 0.16
    # 生理微颤幅度（em 比例）
    tremor: float = 0.0016


@dataclass(frozen=True)
class Drift:
    """状态漂移：人写长文时手会变。低频相关，不是逐字独立随机。"""

    # AR(1) 相关系数：越接近 1 漂移越缓慢
    rho: float = 0.93
    # 各维度的稳态标准差
    # 2026-08-28 素材整页 125 张能量到字号漂移的中位数（原默认 0.042）
    size_sigma: float = 0.106
    slant_sigma: float = 0.022
    baseline_sigma: float = 0.020
    # 无横线时行距由手控制，不由纸控制。真人写不出等距的行 —— 完全均匀的
    # 行距是合成页最容易被认出来的特征之一。相对行距的标准差。
    line_pitch_sigma: float = 0.045
    speed_sigma: float = 0.14
    pressure_sigma: float = 0.075
    # 疲劳趋势：整篇线性变化量（末尾相对开头）
    fatigue_size: float = 0.035
    fatigue_slant: float = 0.014
    fatigue_speed: float = 0.10


@dataclass(frozen=True)
class Pen:
    """笔具。宽度单位是毫米，由 px/mm 换算到像素（开发原则 #3）。"""

    pen_id: str = "gel-0.5"
    family: str = "gel"  # gel | ballpoint | fountain | pencil | marker
    # 落在纸上的**墨线宽度**，不是笔珠直径。0.5mm 中性笔的墨线约 0.44mm。
    # 对照 g8 笔记本语料标定：真人笔画中位 3.0px（1600px 页高），此前取 0.38
    # 只有 2.0px —— 那个值是按 g7 方格答题卡定的，而那批的"笔宽"混进了拍照
    # 背景与桌面阴影，把真人笔画统计整体拉粗了，属于被坏数据带偏的过度修正。
    nib_width_mm: float = 0.44
    # 笔尖足迹的椭圆率（1.0 = 正圆）
    nib_aspect: float = 1.0
    # 笔尖长轴方位角（弧度），决定横细竖粗的方向性
    nib_azimuth: float = 0.0
    # 压力→宽度增益：r ∝ (1 + gain·P)^(1/3)
    pressure_width_gain: float = 0.62
    # 速度→宽度损失
    speed_width_loss: float = 0.14
    # 单位路程的墨量沉积基准
    base_flow: float = 1.0
    pressure_flow_gain: float = 0.42
    # 供墨储备与恢复（断墨/飞白）
    reservoir: float = 1.0
    reservoir_recovery: float = 0.85
    starvation: float = 0.10
    # 纸纤维洇散（像素，@300dpi 基准）
    bleed_sigma_px: float = 0.42
    # 边缘咖啡环强度
    edge_ring: float = 0.16
    # 低速/转折处的墨水堆积强度
    pooling: float = 0.22
    # 墨色（sRGB 0-255）与分通道吸收系数
    ink_rgb: tuple[int, int, int] = (26, 32, 58)
    absorb_rgb: tuple[float, float, float] = (1.05, 1.0, 0.86)


@dataclass(frozen=True)
class WriterStyle:
    """完整的书写者风格。"""

    style_id: str
    display_name: str
    structure: Structure = field(default_factory=Structure)
    brush: Brushwork = field(default_factory=Brushwork)
    motion: Motion = field(default_factory=Motion)
    drift: Drift = field(default_factory=Drift)
    pen: Pen = field(default_factory=Pen)

    def with_pen(self, pen: Pen) -> "WriterStyle":
        return replace(self, pen=pen)


# ---------------------------------------------------------------- 笔具预设

PENS: dict[str, Pen] = {
    "gel-0.5": Pen(),
    "gel-0.38": Pen(
        pen_id="gel-0.38",
        nib_width_mm=0.29,
        pressure_width_gain=0.55,
        bleed_sigma_px=0.34,
        pooling=0.18,
    ),
    "ballpoint-0.7": Pen(
        pen_id="ballpoint-0.7",
        family="ballpoint",
        nib_width_mm=0.52,
        pressure_width_gain=0.48,
        speed_width_loss=0.20,
        base_flow=0.82,
        starvation=0.24,
        reservoir_recovery=0.62,
        bleed_sigma_px=0.22,
        edge_ring=0.08,
        pooling=0.30,
        ink_rgb=(30, 40, 78),
        absorb_rgb=(1.10, 1.0, 0.78),
    ),
    "fountain-f": Pen(
        pen_id="fountain-f",
        family="fountain",
        nib_width_mm=0.46,
        nib_aspect=1.55,
        nib_azimuth=math.radians(-42.0),
        pressure_width_gain=0.86,
        speed_width_loss=0.10,
        base_flow=1.18,
        pressure_flow_gain=0.55,
        starvation=0.06,
        bleed_sigma_px=1.15,
        edge_ring=0.30,
        pooling=0.38,
        ink_rgb=(18, 24, 52),
        absorb_rgb=(1.12, 1.0, 0.72),
    ),
    "pencil-hb": Pen(
        pen_id="pencil-hb",
        family="pencil",
        nib_width_mm=0.68,
        pressure_width_gain=0.70,
        speed_width_loss=0.06,
        base_flow=0.62,
        pressure_flow_gain=0.72,
        starvation=0.30,
        bleed_sigma_px=0.30,
        edge_ring=0.04,
        pooling=0.06,
        ink_rgb=(58, 58, 62),
        absorb_rgb=(1.0, 1.0, 1.0),
    ),
}


# ---------------------------------------------------------------- 风格预设

def _preset(
    style_id: str,
    display_name: str,
    *,
    structure: Structure,
    brush: Brushwork,
    motion: Motion,
    drift: Drift | None = None,
    pen: str = "gel-0.5",
) -> WriterStyle:
    return WriterStyle(
        style_id=style_id,
        display_name=display_name,
        structure=structure,
        brush=brush,
        motion=motion,
        drift=drift or Drift(),
        pen=PENS[pen],
    )


PRESETS: dict[str, WriterStyle] = {
    # 工整楷书：笔笔断开，提按明显，结构端正
    "kai-neat": _preset(
        "kai-neat",
        "工整楷书",
        structure=Structure(
            slant=0.020, heng_rise=0.040, zhonggong=0.985, aspect=1.0,
            component_ratio=-0.020, elastic=0.014,
        ),
        brush=Brushwork(
            entry_hidden=0.80, exit_pressed=0.72, corner_radius=0.018, tiba_depth=0.40,
            curvature_bias=0.018,
        ),
        motion=Motion(
            mean_speed=3.6, speed_cv=0.18, ligature_propensity=0.02, junction_pause_s=0.085,
            tremor=0.0012,
        ),
        drift=Drift(size_sigma=0.026, slant_sigma=0.012, fatigue_size=0.018),
    ),
    # 日常行楷：最常用的"普通人字迹"
    "xingkai-daily": _preset("xingkai-daily", "日常行楷", structure=Structure(), brush=Brushwork(), motion=Motion()),
    # 快写行书：连笔多，速度快，结构松
    "xingshu-fast": _preset(
        "xingshu-fast",
        "快写行书",
        structure=Structure(
            slant=0.105, heng_rise=0.082, zhonggong=0.925, aspect=1.0,
            component_jitter=0.026, elastic=0.034,
        ),
        brush=Brushwork(
            entry_hidden=0.28, exit_pressed=0.26, exit_taper=0.046, corner_radius=0.048,
            tiba_depth=0.30, curvature_bias=0.052,
        ),
        motion=Motion(
            mean_speed=8.4, speed_cv=0.34, ligature_propensity=0.68, ligature_distance=0.46,
            junction_pause_s=0.020, tremor=0.0021,
        ),
        drift=Drift(size_sigma=0.058, slant_sigma=0.032, fatigue_size=0.055, fatigue_speed=0.16),
    ),
    # 学生体：字偏大偏圆，提按弱，略稚拙
    "student": _preset(
        "student",
        "学生体",
        structure=Structure(
            slant=0.012, heng_rise=0.030, zhonggong=1.045, aspect=1.04, component_ratio=0.010,
            component_jitter=0.024, elastic=0.030,
        ),
        brush=Brushwork(
            entry_hidden=0.45, exit_pressed=0.55, corner_radius=0.038, tiba_depth=0.22,
            curvature_bias=0.040,
        ),
        motion=Motion(
            mean_speed=3.1, speed_cv=0.30, ligature_propensity=0.08, junction_pause_s=0.070,
            tremor=0.0026,
        ),
        drift=Drift(size_sigma=0.055, baseline_sigma=0.030, fatigue_size=0.048),
        pen="gel-0.38",
    ),
    # 潦草速记：难认，连笔极多
    "scrawl": _preset(
        "scrawl",
        "潦草速记",
        structure=Structure(
            slant=0.155, heng_rise=0.098, zhonggong=0.885, aspect=1.0,
            component_jitter=0.036, elastic=0.046,
        ),
        brush=Brushwork(
            entry_hidden=0.15, exit_pressed=0.14, exit_taper=0.055, corner_radius=0.062,
            tiba_depth=0.26, curvature_bias=0.070,
        ),
        motion=Motion(
            mean_speed=11.0, speed_cv=0.42, ligature_propensity=0.86, ligature_distance=0.55,
            junction_pause_s=0.012, tremor=0.0030,
        ),
        drift=Drift(size_sigma=0.072, slant_sigma=0.042, fatigue_size=0.070),
        pen="ballpoint-0.7",
    ),
    # 钢笔正楷：适合展示墨迹物理（洇散、咖啡环）
    "fountain-formal": _preset(
        "fountain-formal",
        "钢笔正楷",
        structure=Structure(
            slant=0.038, heng_rise=0.050, zhonggong=0.955, aspect=1.0, elastic=0.018
        ),
        brush=Brushwork(entry_hidden=0.72, exit_pressed=0.66, tiba_depth=0.46, curvature_bias=0.024),
        motion=Motion(mean_speed=4.2, speed_cv=0.22, ligature_propensity=0.14, tremor=0.0014),
        drift=Drift(size_sigma=0.042),
        pen="fountain-f",
    ),
}

DEFAULT_STYLE_ID = "xingkai-daily"

_CUSTOM: dict[str, WriterStyle] = {}


def register_style(style: WriterStyle) -> None:
    _CUSTOM[style.style_id] = style


def has_style(style_id: str) -> bool:
    return style_id in PRESETS or style_id in _CUSTOM


def _with_legacy_xingkai(style: WriterStyle) -> WriterStyle:
    """把日常行楷的三项可测参数还原到素材标定前。其它预设不动。"""
    if style.style_id != DEFAULT_STYLE_ID:
        return style
    return replace(
        style,
        structure=replace(
            style.structure,
            slant=_LEGACY_XINGKAI_SLANT,
            aspect=_LEGACY_XINGKAI_ASPECT,
        ),
        drift=replace(style.drift, size_sigma=_LEGACY_XINGKAI_SIZE_SIGMA),
    )


def get_style(style_id: str | None) -> WriterStyle:
    if not style_id:
        style = PRESETS[DEFAULT_STYLE_ID]
    elif style_id in PRESETS:
        style = PRESETS[style_id]
    elif style_id in _CUSTOM:
        style = _CUSTOM[style_id]
    else:
        raise KeyError("unknown style: " + repr(style_id))
    if _legacy.USE_LEGACY_STYLE:
        return _with_legacy_xingkai(style)
    return style


def list_styles() -> list[dict]:
    styles = list(PRESETS.values()) + list(_CUSTOM.values())
    return [{"id": s.style_id, "name": s.display_name, "pen": s.pen.pen_id} for s in styles]
