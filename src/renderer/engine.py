"""渲染引擎适配层。

- ``RenderEngine``：最小协议（``render(text, config) -> RenderResult``）。
- ``HandrightEngine``：字体填充基线（``engine="handright"``）。快且确定，用作
  单元测试默认引擎与应急回滚，不是产品路径。
- ``V3Engine``：生产路径，见 :mod:`renderer.v3_engine`，按需导入。

``core.render()`` 根据 ``config.engine`` 调度；引擎返回**逻辑分辨率**的页面图像
列表，后处理（滤镜 / preview 缩小 / 格式输出）统一由 ``core.render()`` 负责。
"""

from dataclasses import dataclass
from typing import List, Optional, Protocol

from PIL import Image

from . import core


@dataclass(frozen=True)
class RenderResult:
    """渲染结果：逻辑分辨率（已按 antialias 采样回缩）的页面图像列表。"""

    pages: List[Image.Image]
    # 排完全文实际需要的页数。config.max_pages 截页时会大于 len(pages)，
    # 让上层能照实告诉用户"共 N 页，这里只预览前几页"。None = 等于 len(pages)。
    total_pages: Optional[int] = None


class RenderEngine(Protocol):
    def render(self, text: str, config: "core.RenderConfig") -> RenderResult: ...


class HandrightEngine:
    """现有 handrightbeta 渲染逻辑的封装（默认引擎，``engine="handright"``）。"""

    def render(self, text: str, config: "core.RenderConfig") -> RenderResult:
        layout = core._resolve_layout(config)
        factor = core._supersample_factor(config)

        # 逻辑分辨率字体用于缺字检测与预换行（超采样为 1x 时即渲染字体）
        logical_font, font_path = core._load_font_for(layout.font_size, config.font_id)
        missing = core._check_font_coverage(text, logical_font, font_path)
        if missing:
            raise ValueError(core._missing_message(missing))

        segments = core._prepare_text(text, config, layout, logical_font)

        render_layout = core._scale_layout(layout, factor)
        render_font = (
            logical_font
            if factor == 1
            else core._load_font_for(render_layout.font_size, config.font_id)[0]
        )
        pages = core._build_pages(segments, config, render_layout, render_font, scale=factor)

        # 超采样：渲染 Nx 后 LANCZOS 回缩到逻辑分辨率（抗锯齿）
        if factor != 1:
            target = (layout.page_width, layout.page_height)
            pages = [p.resize(target, Image.Resampling.LANCZOS) for p in pages]
        return RenderResult(pages=pages)


_ENGINES = {
    "handright": HandrightEngine(),
}


def get_engine(name: str) -> RenderEngine:
    """按名称返回引擎实例；未知名称抛 ValueError。"""
    if name == "v3":
        from .v3_engine import V3Engine
        return V3Engine()
    try:
        return _ENGINES[name]
    except KeyError:
        raise ValueError(f"unsupported engine: {name}") from None
