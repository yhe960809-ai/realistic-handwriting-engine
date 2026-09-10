"""手写如真渲染包。

两个引擎，各有明确职责：

- ``v3``：生产路径。骨架轨迹 + 墨迹物理 + 成像仿真，产品发的每一张图都走它。
- ``handright``：字体填充基线。快、确定，用作单元测试的默认引擎与应急回滚。

历史上的 ``multifont`` / ``r2-correlated`` / ``r2-glyphbank`` / ``r2-personal`` /
``stroke-v2`` 全部基于"字体轮廓 + 随机扰动"，与 v3 不是同一个抽象层，两年没有
进过产品，已删除（git 历史保留）。
"""

from .core import (
    RenderConfig,
    RenderedDocument,
    page_size,
    render,
    render_document,
    render_with_metadata,
)
from .engine import HandrightEngine, RenderEngine, RenderResult

__all__ = [
    "render",
    "render_with_metadata",
    "render_document",
    "RenderedDocument",
    "RenderConfig",
    "page_size",
    "RenderEngine",
    "RenderResult",
    "HandrightEngine",
]
