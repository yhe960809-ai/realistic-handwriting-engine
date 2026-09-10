"""渲染核心实现。

E1-B-3（G1 收口）：
- 手动排版标记：``---`` 分页、``>>>`` 右对齐
- 多格式输出：PNG / JPG / 多页 PDF（ReportLab）
- 字体缺字检测（fontTools cmap）
- preview / hd 布局一致
- 安全限制：输入长度、页数、图片尺寸

E1-C（视觉真实性专项优化）：
- 纸张预设 PagePreset（a4 / notebook / letter，默认 a4，210:297 比例）
- 混合中英文换行（英文整词不拆，中文标点避头尾）
- 克制的手写不规则度（handrightbeta 扰动参数，seed 可控复现）
- 墨水浓淡变化（``ink_variation``，默认黑色中性笔观感）
- 程序化纸张背景（``paper_style``：plain / lined / grid，纯代码生成不下载图片）
- 段落首行缩进（``paragraph_indent``，默认 2 个全角字宽，真实排版而非空格填充）
"""

import functools
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from handright import Template, handwrite
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
import reportlab.rl_config
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# ReportLab 默认把图片流做 ASCII85 编码。它的 C 加速模块（_rl_accel）在多数
# 环境里没装，纯 Python 的 base85 编码在整页位图上要跑好几秒 —— 一份 2 页 A4
# 的 PDF 里，光这一步就占了三成耗时。PDF 完全支持二进制流，关掉 A85 即可，
# 图片仍走 Flate 压缩，输出依然确定性可复现。
reportlab.rl_config.useA85 = 0

from . import aigc, fonts

# 安全上限
MAX_TEXT_LENGTH = 10_000
MAX_PAGE_WIDTH = 3_200
MAX_PAGE_HEIGHT = 4_500
MAX_PAGES = 50

# PDF 内嵌 JPEG 的质量。subsampling=0（4:4:4）保留笔画边缘的色度，
# 92 在 254 DPI 打印下与无损无可见差别。
_PDF_JPEG_QUALITY = 92

# preview 相对 hd 的缩放比例。preview 与 hd 使用相同的 seed 与排版参数，
# 仅在 hd 渲染完成后等比缩小，从而保证布局、换行、分页完全一致。
PREVIEW_SCALE = 0.5

# 中文标点避头尾（供 handright 内部换行使用）
_START_CHARS = "，。、；：？！）〕》」』\"\"''〉〗】）]}>,.?!:;"
_END_CHARS = "（〔《「『\"\"''〈〖【（[{"

# 换行层面的避头/避尾集合（我方预换行使用）
# 避头：这些字符不得出现在一行的开头
_AVOID_LINE_START = "，。、；：？！）》】”’…—.,!?;:)]}"
# 避尾：这些字符不得出现在一行的末尾
_AVOID_LINE_END = "（《【“‘([{"

# 中文字体候选路径（按优先级）
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf", # 黑体
    "C:/Windows/Fonts/simsun.ttc", # 宋体
]

# 纸张预设（逻辑坐标，preview 与 hd 共用同一坐标系，仅输出尺寸不同）
# a4 按 210:297 比例 ×10；每行约 23 字、每页约 24 行，1000 字约 2 页。
_PAGE_PRESETS: Dict[str, Dict[str, int]] = {
    "a4": {
        "page_width": 2100,
        "page_height": 2970,
        "margin": 150,
        "font_size": 75,
        "line_spacing": 110,
    },
    "notebook": {
        "page_width": 1760,
        "page_height": 2480,
        "margin": 130,
        "font_size": 64,
        "line_spacing": 96,
    },
    "letter": {
        "page_width": 2160,
        "page_height": 2794,
        "margin": 150,
        "font_size": 75,
        "line_spacing": 110,
    },
}

# 墨水基色：深灰而非纯黑，接近黑色中性笔落在纸上的观感
_INK_FILL = (35, 35, 35)

# 笔颜色（hex）校验：^#[0-9a-fA-F]{6}$（默认 #232323 == _INK_FILL）
_PEN_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_FONT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# 抗锯齿超采样倍数（preview 模式恒为 1x，用较低成本）
_SUPERSAMPLE_FACTORS = {"off": 1, "standard": 2, "high": 3}

# 引擎 / 枚举合法值
_ENGINE_NAMES = ("handright", "v3")
_ANTIALIAS_VALUES = ("off", "standard", "high")
_FILTER_STYLES = ("none", "soft", "scan")

# 手写不规则度（克制）：相对 handrightbeta 默认值略增强，避免「自动排版」感
_HW_LINE_SPACING_SIGMA = 1 / 24   # 行内字符纵向抖动
_HW_FONT_SIZE_SIGMA = 1 / 36      # 字大小变化
_HW_WORD_SPACING_SIGMA = 1 / 24   # 字间距变化
_HW_PERTURB_X_SIGMA = 1 / 24      # 笔画横向扰动
_HW_PERTURB_Y_SIGMA = 1 / 24      # 笔画纵向扰动
_HW_PERTURB_THETA_SIGMA = 0.06    # 笔画旋转（弧度，约 3.4°）

# 程序化纸张背景线颜色（淡灰，不干扰墨迹判定）
_LINED_COLOR = (226, 226, 226)
_GRID_COLOR = (232, 232, 232)

# 分词正则：英文/数字连续串为一个 token，其余（汉字/标点/全角空格）按单字符切分。
# 注意：这里只用 ASCII 空白 ``[ \\t]`` 作为词间分隔，不能使用 ``\\s``——
# 全角空格 ``\\u3000`` 属于 Unicode 空白，但它是有真实字宽的排版字符
# （段首缩进/右对齐靠它推进），绝不能被折叠成 ASCII 空格。
_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[ \t]+|[^ \tA-Za-z0-9]")


@dataclass(frozen=True)
class RenderConfig:
    """渲染配置。

    ``preset`` 决定纸张比例与默认排版密度；``page_width`` 等显式字段（非 None）
    会覆盖 preset 的对应默认值。preview 与 hd 共用同一逻辑坐标系。
    """

    preset: str = "a4"
    seed: int = 42
    format: Literal["png", "jpg", "pdf"] = "png"
    mode: Literal["preview", "hd"] = "hd"
    paper_style: Literal["plain", "lined", "grid"] = "plain"
    background_id: str = ""            # 真实背景图 id（assets/backgrounds/）；""=程序化纸张
    paragraph_indent: int = 2          # 段首缩进的字符数（0 = 不缩进）
    ink_variation: float = 0.1         # 墨水浓淡变化强度（0.0 ~ 1.0）

    # 排版缩放（相对 preset；1.0 = 不变，用于「字体大小 / 行间距 / 页边距」三档）
    font_size_scale: float = 1.0
    line_spacing_scale: float = 1.0
    margin_scale: float = 1.0
    # 基础字间距（像素，可为负 = 更紧凑；0 = 默认）
    word_spacing: int = 0

    # 手写自然度（相对字号的比值；默认值 = E1-C 常量，保证旧行为不变）
    perturb_x_ratio: float = _HW_PERTURB_X_SIGMA        # 笔画横向浮动
    perturb_y_ratio: float = _HW_PERTURB_Y_SIGMA        # 笔画纵向浮动
    font_size_ratio: float = _HW_FONT_SIZE_SIGMA        # 字号随机波动
    word_spacing_ratio: float = _HW_WORD_SPACING_SIGMA  # 字间距随机波动
    line_spacing_ratio: float = _HW_LINE_SPACING_SIGMA  # 行间距随机波动
    perturb_theta_sigma: float = _HW_PERTURB_THETA_SIGMA  # 笔画旋转（弧度，绝对值）

    # 显式覆盖 preset 排版参数（None = 使用 preset 默认值）
    page_width: Optional[int] = None
    page_height: Optional[int] = None
    margin: Optional[int] = None
    font_size: Optional[int] = None
    line_spacing: Optional[int] = None

    # R1 新增字段（全部默认值保证与旧 RenderConfig() 逐字节兼容）
    font_id: str = ""                               # registry id；""=系统字体回退
    margin_top: Optional[int] = None                # 独立上边距（px）
    margin_bottom: Optional[int] = None             # 独立下边距（px）
    margin_left: Optional[int] = None               # 独立左边距（px）
    margin_right: Optional[int] = None              # 独立右边距（px）
    antialias: Literal["off", "standard", "high"] = "off"   # 1x/2x/3x 超采样
    filter_style: Literal["none", "soft", "scan"] = "none"  # 后处理滤镜
    pen_color: str = "#232323"                      # 笔颜色 hex（#232323 == _INK_FILL）
    # v3 笔具。空 = 用风格自带笔；只允许 renderer.v3.style.PENS 里的 id。
    pen_id: str = ""
    preview_width: Optional[int] = None             # 仅 mode=preview 生效（px）
    # 引擎调度键。生产走 "v3"；"handright" 是字体填充基线，快而确定，
    # 保留给单元测试与应急回滚（API 层的 setdefault 会把产品请求置为 v3）。
    engine: Literal["handright", "v3"] = "handright"
    # v3 的书写风格 id（WSM）。内置预设或 "fit-*" 用户专属风格。
    stroke_writer_id: str = "xingkai-daily"

    # 只渲染前 N 页（预览用）。None = 全部。排版仍按全文计算，因此分页、
    # 换行与笔迹漂移都与完整输出一致，只是不把后面的页画出来。
    max_pages: Optional[int] = None

    # AIGC 生成合成内容标识（《标识办法》/ GB 45438-2025）。
    # 空 producer = 不写任何标识：渲染器作为库被直接调用时不冒充服务提供者。
    # 对外服务必须由 API 层显式注入 producer（见 api.config.aigc_producer）。
    aigc_producer: str = ""
    aigc_visible: bool = True                       # 显式标识（页面可见提示）
    # 多块版式（页 → 块列表）。None = 旧的整页回流。块坐标为页百分比。
    layout_pages: Optional[tuple] = None
    text_align: str = "left"
    # v3 整页书写倾斜（弧度，右倾为正）。handright 路径忽略此字段。
    text_slant: float = 0.0
    # 写错再涂改。0 = 关闭，必须与旧页逐位一致。
    scribble_count: int = 0
    scribble_style: str = "strike"
    # 交付形态。page = 现网（纸纹 / 成像 / 照片合成）；plate = 套打白底墨迹。
    # 默认 page，与未设此字段时逐字节一致。
    output_mode: Literal["page", "plate"] = "page"


@dataclass(frozen=True)
class _Layout:
    """解析后的具体排版参数（四边距独立）。"""

    page_width: int
    page_height: int
    margin_top: int
    margin_bottom: int
    margin_left: int
    margin_right: int
    font_size: int
    line_spacing: int

    @property
    def content_width(self) -> int:
        return self.page_width - self.margin_left - self.margin_right

    @property
    def content_height(self) -> int:
        return self.page_height - self.margin_top - self.margin_bottom


def _resolve_layout(cfg: RenderConfig) -> _Layout:
    """将 preset 与显式覆盖合并为具体排版参数。

    四边距解析优先级（契约 §1，向后兼容关键）：:

        base = preset.margin
        if margin is not None: base = margin
        top    = margin_top    if margin_top    is not None else base
        bottom = margin_bottom if margin_bottom is not None else base
        left   = margin_left   if margin_left   is not None else base
        right  = margin_right  if margin_right  is not None else base

    四个新字段全为 None 时，等价于旧 ``margin`` + ``margin_scale``，逐字节不变。
    """
    preset = _PAGE_PRESETS.get(cfg.preset)
    if preset is None:
        raise ValueError(f"unknown preset: {cfg.preset}")

    base_margin = cfg.margin if cfg.margin is not None else preset["margin"]
    margin_top = cfg.margin_top if cfg.margin_top is not None else base_margin
    margin_bottom = cfg.margin_bottom if cfg.margin_bottom is not None else base_margin
    margin_left = cfg.margin_left if cfg.margin_left is not None else base_margin
    margin_right = cfg.margin_right if cfg.margin_right is not None else base_margin

    font_size = cfg.font_size if cfg.font_size is not None else preset["font_size"]
    line_spacing = cfg.line_spacing if cfg.line_spacing is not None else preset["line_spacing"]

    # 排版缩放（默认 1.0 不变；缩放作用于 preset 基准值之上）
    if cfg.margin_scale != 1.0:
        margin_top = int(round(margin_top * cfg.margin_scale))
        margin_bottom = int(round(margin_bottom * cfg.margin_scale))
        margin_left = int(round(margin_left * cfg.margin_scale))
        margin_right = int(round(margin_right * cfg.margin_scale))
    if cfg.font_size_scale != 1.0:
        font_size = int(round(font_size * cfg.font_size_scale))
    if cfg.line_spacing_scale != 1.0:
        line_spacing = int(round(line_spacing * cfg.line_spacing_scale))

    return _Layout(
        page_width=cfg.page_width if cfg.page_width is not None else preset["page_width"],
        page_height=cfg.page_height if cfg.page_height is not None else preset["page_height"],
        margin_top=margin_top,
        margin_bottom=margin_bottom,
        margin_left=margin_left,
        margin_right=margin_right,
        font_size=font_size,
        line_spacing=line_spacing,
    )


def page_size(config: Optional[RenderConfig] = None) -> Tuple[int, int]:
    """返回渲染后的逻辑页面尺寸（preview 模式下为缩放前尺寸）。"""
    layout = _resolve_layout(config or RenderConfig())
    return layout.page_width, layout.page_height


def render(text: str, config: Optional[RenderConfig] = None) -> bytes:
    """将文本渲染为手写风格图片或 PDF。

    支持 ``---`` 手动分页和 ``>>>`` 右对齐标记。
    长文本自动按配置尺寸分页；多页 PNG 会垂直拼接为长图。

    根据 ``config.engine`` 调度到对应渲染引擎（默认 ``handright``，向后兼容）。
    引擎返回逻辑分辨率页面后，统一走滤镜 -> preview 缩小 -> 格式输出的后处理。

    Args:
        text: 输入文本。
        config: 渲染配置，为 None 时使用默认配置（a4 / png / hd）。

    Returns:
        PNG 图片字节 或 多页 PDF 字节。
    """
    return render_with_metadata(text, config)[0]


class RenderedDocument:
    """一次排版的结果，可以按多种格式反复编码。

    排版是全流程里最贵的一步（v3 的一页 A4 约数秒），而"生成 PDF 同时给一张
    预览图"是产品的常规需求。把排版与编码拆开，这类需求就只排版一次：

        doc = render_document(text, hd_config)
        pdf, _ = doc.encode(fmt="pdf")
        png, _ = doc.encode(fmt="png", preview_width=800)

    ``pages`` 是引擎输出、已过滤镜的逻辑分辨率页面，尚未缩放、未合成照片背景、
    未打 AIGC 标识 —— 这三步与目标格式有关，放在 :meth:`encode` 里。
    """

    __slots__ = ("pages", "config", "text", "total_pages")

    def __init__(
        self,
        pages: List[Image.Image],
        config: RenderConfig,
        text: str,
        total_pages: Optional[int] = None,
    ):
        self.pages = pages
        self.config = config
        self.text = text
        # 全文实际页数。截页预览时大于 len(pages)。
        self.total_pages = total_pages if total_pages is not None else len(pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def truncated(self) -> bool:
        return self.total_pages > len(self.pages)

    def encode(
        self,
        fmt: Optional[str] = None,
        preview_width: Optional[int] = None,
    ) -> Tuple[bytes, Dict[str, object]]:
        """编码为字节。``preview_width`` 非空时按该宽度等比缩小。"""
        cfg = self.config
        fmt = fmt or cfg.format
        pages = self.pages

        # preview：在完整排版后等比缩小，保证与 hd 的换行、分页完全一致。
        if preview_width is not None:
            scale = preview_width / pages[0].width
            if abs(scale - 1.0) > 0.02:
                pages = [_downscale(page, scale) for page in pages]
        elif cfg.mode == "preview":
            scale = _preview_scale(cfg, pages[0].width)
            if abs(scale - 1.0) > 0.02:
                pages = [_downscale(page, scale) for page in pages]

        # 照片背景是"两阶段渲染"：引擎先在规范化白纸坐标排版，缩放也只处理
        # 纸面墨迹；最后才把墨迹透视映射到同分辨率照片。合成放在缩放之后，
        # 避免 LANCZOS 把边界墨迹扩散到纸面外，保证四边形外像素严格不变。
        if cfg.background_id and not is_plate(cfg):
            pages = [_composite_page_on_photo_surface(page, cfg) for page in pages]

        # AIGC 标识是输出链路的最后一步：显式标识画在合成之后（否则会被透视
        # 映射扭曲），隐式标识随各格式的编码器写入。
        label = _resolve_aigc_label(self.text, cfg)
        if label is not None and cfg.aigc_visible:
            pages = [aigc.draw_visible_label(page, label) for page in pages]

        if fmt == "pdf":
            output = _images_to_pdf(
                pages,
                label,
                photographic=_is_photographic(cfg),
                page_size_pt=_plate_page_size_pt(cfg) if is_plate(cfg) else None,
            )
            raster = pages[0]
        else:
            combined = pages[0] if len(pages) == 1 else _concat_pages(pages)
            raster = combined
            if fmt == "jpg":
                output = _image_to_jpg(combined, label=label)
            else:
                output = _image_to_bytes(combined, label=label)

        metadata: Dict[str, object] = {
            "pages": len(pages),
            "total_pages": self.total_pages,
            "truncated": self.truncated,
            "width": raster.width,
            "height": raster.height,
            "bytes": len(output),
        }
        if label is not None:
            metadata["aigc"] = {**label.to_dict(), "visible": bool(cfg.aigc_visible)}
        return output, metadata


def render_document(text: str, config: Optional[RenderConfig] = None) -> RenderedDocument:
    """只排版，不编码。用于同一份排版要出多种格式的场景。"""
    if not isinstance(text, str):
        raise TypeError("text must be str")

    cfg = config or RenderConfig()
    layout = _resolve_layout(cfg)
    _validate_input(cfg, layout, text)

    from .engine import get_engine

    engine = get_engine(cfg.engine)
    result = engine.render(text, cfg)

    all_pages: List[Image.Image] = list(result.pages)
    if not all_pages:
        all_pages = [_make_background(cfg, layout)]

    total_pages = result.total_pages or len(all_pages)

    # 未原生支持截页的引擎在这里统一截断。它们已经把所有页都画完了，省不了
    # 时间，但至少输出语义一致（v3 是产品路径，它在排版阶段就真的少画了）。
    if cfg.max_pages is not None and len(all_pages) > cfg.max_pages:
        all_pages = all_pages[: cfg.max_pages]

    if total_pages > MAX_PAGES:
        raise ValueError(
            f"too many pages generated: {total_pages} (max {MAX_PAGES})"
        )

    # 后处理滤镜（逻辑分辨率，默认 none = 不处理，保证向后兼容）。
    # v3 的成像仿真已经覆盖 filter_style 的语义，再叠一层 PIL 滤镜是双重劣化。
    if cfg.filter_style != "none" and cfg.engine != "v3":
        all_pages = [_apply_filter(page, cfg.filter_style) for page in all_pages]

    return RenderedDocument(all_pages, cfg, text, total_pages=total_pages)


def render_with_metadata(
    text: str, config: Optional[RenderConfig] = None
) -> Tuple[bytes, Dict[str, object]]:
    """渲染并附带运行元数据（页数 / 输出尺寸），供 API 可观测性日志使用。

    输出字节与 :func:`render` 完全一致；metadata 不含任何用户文本内容：
    ``{"pages": 页数, "width": 输出像素宽, "height": 输出像素高, "bytes": 字节数}``。
    """
    return render_document(text, config).encode()


def is_plate(cfg: RenderConfig) -> bool:
    """套打底板：白底墨迹，不进纸纹、成像和照片合成。"""
    return getattr(cfg, "output_mode", "page") == "plate"


def _is_photographic(cfg: RenderConfig) -> bool:
    """页面是否为连续色调。v3 走成像仿真，真实背景本身就是照片。"""
    if is_plate(cfg):
        return False
    return cfg.engine == "v3" or bool(cfg.background_id)


def _plate_page_size_pt(cfg: RenderConfig) -> Tuple[float, float]:
    """套打 PDF 的物理页框。逻辑像素按 10 px/mm（preset 与兼容层同一约定）。"""
    layout = _resolve_layout(cfg)
    return (layout.page_width * 72.0 / 254.0, layout.page_height * 72.0 / 254.0)


def _resolve_aigc_label(text: str, cfg: RenderConfig) -> Optional["aigc.AigcLabel"]:
    """没有配置服务提供者时不打标识（库调用），否则确定性导出。"""
    if not cfg.aigc_producer:
        return None
    return aigc.build_label(text, cfg, cfg.aigc_producer)


def _validate_input(cfg: RenderConfig, layout: _Layout, text: str) -> None:
    """校验输入与配置是否在安全范围内。"""
    from .layout import MAX_TEXT_LENGTH as LAYOUT_MAX_TEXT

    text_limit = LAYOUT_MAX_TEXT if cfg.layout_pages else MAX_TEXT_LENGTH
    if len(text) > text_limit:
        raise ValueError(
            f"text too long: {len(text)} characters (max {text_limit})"
        )
    if layout.page_width > MAX_PAGE_WIDTH or layout.page_height > MAX_PAGE_HEIGHT:
        raise ValueError(
            f"page size too large: {layout.page_width}x{layout.page_height} "
            f"(max {MAX_PAGE_WIDTH}x{MAX_PAGE_HEIGHT})"
        )
    if cfg.format not in ("png", "jpg", "pdf"):
        raise ValueError(f"unsupported format: {cfg.format}")
    if cfg.mode not in ("preview", "hd"):
        raise ValueError(f"unsupported mode: {cfg.mode}")
    if cfg.paper_style not in ("plain", "lined", "grid"):
        raise ValueError(f"unsupported paper_style: {cfg.paper_style}")
    if cfg.preset not in _PAGE_PRESETS:
        raise ValueError(f"unsupported preset: {cfg.preset}")
    if not 0 <= cfg.paragraph_indent <= 8:
        raise ValueError(f"paragraph_indent out of range: {cfg.paragraph_indent}")
    if not 0.0 <= cfg.ink_variation <= 1.0:
        raise ValueError(f"ink_variation out of range: {cfg.ink_variation}")

    # 专业参数安全范围（产品安全上限，非 handrightbeta 技术极限）
    if not 0.7 <= cfg.font_size_scale <= 1.4:
        raise ValueError(f"font_size_scale out of range: {cfg.font_size_scale}")
    if not 0.7 <= cfg.line_spacing_scale <= 1.6:
        raise ValueError(f"line_spacing_scale out of range: {cfg.line_spacing_scale}")
    if not 0.5 <= cfg.margin_scale <= 1.8:
        raise ValueError(f"margin_scale out of range: {cfg.margin_scale}")
    if not -30 <= cfg.word_spacing <= 30:
        raise ValueError(f"word_spacing out of range: {cfg.word_spacing}")
    for name, value in (
        ("perturb_x_ratio", cfg.perturb_x_ratio),
        ("perturb_y_ratio", cfg.perturb_y_ratio),
        ("font_size_ratio", cfg.font_size_ratio),
        ("word_spacing_ratio", cfg.word_spacing_ratio),
        ("line_spacing_ratio", cfg.line_spacing_ratio),
    ):
        if not 0.0 <= value <= 0.2:
            raise ValueError(f"{name} out of range: {value}")
    if not 0.0 <= cfg.perturb_theta_sigma <= 0.2:
        raise ValueError(f"perturb_theta_sigma out of range: {cfg.perturb_theta_sigma}")
    if not -0.25 <= cfg.text_slant <= 0.25:
        raise ValueError(f"text_slant out of range: {cfg.text_slant}")

    # R1 新增字段校验
    if cfg.engine not in _ENGINE_NAMES:
        raise ValueError(f"unsupported engine: {cfg.engine}")
    if _FONT_ID_RE.fullmatch(cfg.stroke_writer_id) is None:
        raise ValueError("stroke_writer_id must be a safe registry id")
    if cfg.antialias not in _ANTIALIAS_VALUES:
        raise ValueError(f"unsupported antialias: {cfg.antialias}")
    if cfg.filter_style not in _FILTER_STYLES:
        raise ValueError(f"unsupported filter_style: {cfg.filter_style}")
    if _PEN_COLOR_RE.match(cfg.pen_color) is None:
        raise ValueError(f"invalid pen_color: {cfg.pen_color}")
    if cfg.pen_id:
        from .v3.style import PENS

        if cfg.pen_id not in PENS:
            raise ValueError(f"unsupported pen_id: {cfg.pen_id}")
    if not 0 <= int(cfg.scribble_count) <= 8:
        raise ValueError(f"scribble_count out of range: {cfg.scribble_count}")
    if cfg.scribble_style not in ("strike", "scribble", "cross"):
        raise ValueError(f"unsupported scribble_style: {cfg.scribble_style}")
    if cfg.output_mode not in ("page", "plate"):
        raise ValueError(f"unsupported output_mode: {cfg.output_mode}")
    # 边距按页面尺寸校验，不用固定上限：书写区域（照片模式框选的那一块）就是
    # 用四边距表达的，只写页面下半部分时上边距会接近页高的一半，固定 600px 的
    # 上限会把这种完全正常的请求判成非法。真正的约束是"要剩下能写字的地方"。
    for name, value, extent in (
        ("margin_top", cfg.margin_top, layout.page_height),
        ("margin_bottom", cfg.margin_bottom, layout.page_height),
        ("margin_left", cfg.margin_left, layout.page_width),
        ("margin_right", cfg.margin_right, layout.page_width),
    ):
        if value is not None and not 0 <= value < extent:
            raise ValueError(f"{name} out of range: {value}")
    if layout.content_width < 40 or layout.content_height < 40:
        raise ValueError(
            "writing area too small: "
            f"{layout.content_width}x{layout.content_height}px after margins"
        )
    if cfg.font_size is not None and not 20 <= cfg.font_size <= 300:
        raise ValueError(f"font_size out of range: {cfg.font_size}")
    if cfg.line_spacing is not None and not 20 <= cfg.line_spacing <= 600:
        raise ValueError(f"line_spacing out of range: {cfg.line_spacing}")
    if cfg.preview_width is not None and not 300 <= cfg.preview_width <= 1200:
        raise ValueError(f"preview_width out of range: {cfg.preview_width}")


@functools.lru_cache(maxsize=4)
def _supported_codepoints(font_path: str) -> frozenset:
    """读取字体 cmap，返回受支持字符的码点集合（排除 .notdef）。

    这是判断字体是否支持某字符的可靠依据：FreeType/Pillow 的 ``getmask``
    对不存在的字符不会抛异常，只会回退到 .notdef（方框/tofu glyph），
    无法据此可靠判断缺字。fontTools 直接读取 cmap 表，能准确区分
    「有 glyph」与「无 glyph」。
    """
    from fontTools.ttLib import TTFont

    tt = TTFont(font_path, fontNumber=0)
    cmap = tt.getBestCmap()
    glyph_order = tt.getGlyphOrder()
    notdef_name = glyph_order[0] if glyph_order else ".notdef"

    supported = set()
    for codepoint, glyph_name in cmap.items():
        if glyph_name == notdef_name:
            continue
        try:
            if tt.getGlyphID(glyph_name) == 0:
                continue
        except Exception:
            continue
        supported.add(codepoint)
    return frozenset(supported)


def _check_font_coverage(
    text: str, font: ImageFont.FreeTypeFont, font_path: Path
) -> List[str]:
    """检测字体不支持的字符，返回缺失字符列表（不含空白）。

    优先通过字体 cmap 判断 glyph 覆盖；若 cmap 解析失败，降级为
    getmask 启发式，避免渲染完全阻塞。
    """
    try:
        covered = _supported_codepoints(str(font_path))
    except Exception:
        return _check_font_coverage_via_getmask(text, font)

    missing = [
        char for char in set(text) if not char.isspace() and ord(char) not in covered
    ]
    return sorted(missing)


def _check_font_coverage_via_getmask(
    text: str, font: ImageFont.FreeTypeFont
) -> List[str]:
    """降级启发式：仅能捕获 getmask 抛异常的情况，可靠性有限。"""
    missing: List[str] = []
    for char in set(text):
        if char.isspace():
            continue
        try:
            font.getmask(char)
        except Exception:
            missing.append(char)
    return sorted(missing)


def _missing_message(missing: Sequence[str]) -> str:
    """缺字错误信息（与历史行为一致）。"""
    shown = "".join(missing[:20])
    suffix = "..." if len(missing) > 20 else ""
    return f"font missing {len(missing)} character(s): {shown}{suffix}"


def _parse_pen_color(pen_color: str) -> Tuple[int, int, int]:
    """把 ``#RRGGBB`` 解析为 RGB 三元组（默认 #232323 == _INK_FILL）。"""
    if _PEN_COLOR_RE.match(pen_color) is None:
        raise ValueError(f"invalid pen_color: {pen_color}")
    return (
        int(pen_color[1:3], 16),
        int(pen_color[3:5], 16),
        int(pen_color[5:7], 16),
    )


def _supersample_factor(cfg: RenderConfig) -> int:
    """抗锯齿超采样倍数：off=1x / standard=2x / high=3x；preview 恒 1x（低成本）。"""
    if cfg.mode == "preview":
        return 1
    return _SUPERSAMPLE_FACTORS[cfg.antialias]


def _scale_layout(layout: _Layout, factor: int) -> _Layout:
    """把排版参数整体放大 factor 倍（用于超采样渲染）。"""
    if factor == 1:
        return layout
    return _Layout(
        page_width=layout.page_width * factor,
        page_height=layout.page_height * factor,
        margin_top=layout.margin_top * factor,
        margin_bottom=layout.margin_bottom * factor,
        margin_left=layout.margin_left * factor,
        margin_right=layout.margin_right * factor,
        font_size=layout.font_size * factor,
        line_spacing=layout.line_spacing * factor,
    )


def _prepare_text(
    text: str, cfg: RenderConfig, layout: _Layout, font: ImageFont.FreeTypeFont
) -> List[str]:
    """规范化 → 段首缩进 → 预换行 → 右对齐 → 手动分页，返回各段落文本。"""
    normalized = _normalize_line_breaks(text)
    indented = _apply_paragraph_indent(normalized, cfg.paragraph_indent)
    # 预换行宽度对齐 handright 的实际行容量（content_width - 一个字符宽），
    # 从而由我方控制换行边界，handright 仅按 ``\n`` 分行，不再次拆分英文单词。
    wrapped = _wrap(indented, font, layout.content_width - layout.font_size)
    aligned = _apply_right_align(wrapped, font, layout.content_width)
    return _split_manual_pages(aligned)


def _build_pages(
    segments: Sequence[str],
    cfg: RenderConfig,
    layout: _Layout,
    font: ImageFont.FreeTypeFont,
    scale: int = 1,
) -> List[Image.Image]:
    """对每个分段调用 handrightbeta 生成页面（可能一分为多页）。"""
    all_pages: List[Image.Image] = []
    for segment in segments:
        if not segment.strip():
            continue
        all_pages.extend(_generate_pages(segment, cfg, layout, font, scale))
    return all_pages


def _preview_scale(cfg: RenderConfig, page_width: int) -> float:
    """preview 缩小比例：preview_width 优先，否则缩到 HD 宽 × PREVIEW_SCALE。

    v3 已经按预览分辨率光栅化，这里必须对照 HD 版心宽度算目标，
    不能再乘一次 0.5，否则 1050×1485 会变成 525×742。
    """
    if cfg.preview_width is not None:
        return cfg.preview_width / max(page_width, 1)
    layout_w = _resolve_layout(cfg).page_width
    target = max(1, int(round(layout_w * PREVIEW_SCALE)))
    return target / max(page_width, 1)


def _downscale(image: Image.Image, scale: float) -> Image.Image:
    """按比例等比缩小（LANCZOS）。"""
    new_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _apply_filter(image: Image.Image, filter_style: str) -> Image.Image:
    """后处理滤镜（Step 28）：none 无 / soft 轻微边缘柔化 / scan 极轻微扫描感。"""
    if filter_style == "soft":
        return image.filter(ImageFilter.SMOOTH)
    if filter_style == "scan":
        return _scan_filter(image)
    raise ValueError(f"unsupported filter_style: {filter_style}")


def _scan_filter(image: Image.Image) -> Image.Image:
    """极轻微的纸张/墨迹处理（确定性，无随机噪声）：轻微柔化 + 略微提升对比度。"""
    smooth = image.filter(ImageFilter.SMOOTH)
    return ImageEnhance.Contrast(smooth).enhance(1.03)


def _normalize_line_breaks(text: str) -> str:
    """统一换行符为 LF。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _apply_paragraph_indent(text: str, indent_chars: int) -> str:
    """为每个段落的首行添加全角空格缩进。

    使用全角空格（``\\u3000``）而非 ASCII 空格：其渲染宽度等于一个中文字符
    （getbbox 返回完整字宽且无墨迹），能实现像素级精确的段首缩进。
    仅缩进每个段落的**首行**；``>>>`` 右对齐行与 ``---`` 分页行不缩进。
    """
    if indent_chars <= 0:
        return text
    indent = "\u3000" * indent_chars

    lines = text.split("\n")
    result: List[str] = []
    at_paragraph_start = True

    for line in lines:
        if re.match(r"^\s*-{3,}\s*$", line):
            result.append(line)
            at_paragraph_start = True   # 分页后视为新段落
            continue
        if line.lstrip().startswith(">>>"):
            result.append(line)
            at_paragraph_start = False
            continue
        if line.strip() == "":
            result.append("")
            at_paragraph_start = True
            continue
        if at_paragraph_start:
            result.append(indent + line)
        else:
            result.append(line)
        at_paragraph_start = False

    return "\n".join(result)


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    """按词边界自动换行（英文整词不拆），并对中文标点做避头尾处理。"""
    paragraphs = text.split("\n")
    wrapped_paragraphs: List[str] = []

    for paragraph in paragraphs:
        if not paragraph:
            wrapped_paragraphs.append("")
            continue
        lines = _greedy_wrap(paragraph, font, max_width)
        lines = _fix_punctuation(lines)
        wrapped_paragraphs.append("\n".join(lines))

    return "\n".join(wrapped_paragraphs)


def _greedy_wrap(
    paragraph: str, font: ImageFont.FreeTypeFont, max_width: int
) -> List[str]:
    """词边界贪心换行：英文单词/数字串作为不可拆分单元，仅超宽时逐字符拆。"""
    tokens = _WORD_TOKEN_RE.findall(paragraph)
    lines: List[str] = []
    current = ""

    for tok in tokens:
        if tok[0] in " \t":
            # ASCII 空格/制表符作为词间分隔；行尾空格在换行时被 rstrip 去除。
            # 全角空格 \\u3000 走下面的内容分支，保留其真实字宽。
            if current:
                current += " "
            continue

        if current and font.getlength(current.rstrip() + tok) > max_width:
            lines.append(current.rstrip())
            current = ""

        if len(tok) > 1 and font.getlength(tok) > max_width:
            # 单个英文单词/数字串本身超过行宽，才允许从中间拆开
            for ch in tok:
                if current and font.getlength(current.rstrip() + ch) > max_width:
                    lines.append(current.rstrip())
                    current = ""
                current += ch
        else:
            current += tok

    if current.rstrip():
        lines.append(current.rstrip())
    return lines


def _fix_punctuation(lines: List[str]) -> List[str]:
    """中文标点避头尾：把行首闭标点拉回上一行、行尾开标点推到下一行。"""
    i = 0
    while i < len(lines) - 1:
        cur = lines[i]
        nxt = lines[i + 1]

        # 避尾：行尾不得出现开括号/开引号
        while cur and cur[-1] in _AVOID_LINE_END:
            nxt = cur[-1] + nxt
            cur = cur[:-1]

        # 避头：行首不得出现闭标点，把上一行末字一并下移
        while nxt and nxt[0] in _AVOID_LINE_START and cur:
            nxt = cur[-1] + nxt
            cur = cur[:-1]

        lines[i] = cur
        lines[i + 1] = nxt
        i += 1

    return [ln for ln in lines if ln]


def _apply_right_align(
    text: str, font: ImageFont.FreeTypeFont, content_width: int
) -> str:
    """将 ``>>>文字`` 行替换为全角空格前导的右对齐行。

    使用全角空格（``\\u3000``）而非 ASCII 空格：handright 按 ``getbbox`` 的
    宽度推进每个字符，ASCII 空格仅约 22px，全角空格为完整字符宽度（约等于
    中文字符），才能把文字可靠地推到右边界。缺口按像素宽度计算，兼容
    数字/英文等半角字符混排。
    """
    full_space = "\u3000"
    space_w = font.getlength(full_space) or font.getlength("中") or 1.0

    lines = text.split("\n")
    result: List[str] = []

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(">>>"):
            content = stripped[3:].rstrip()
            gap = content_width - font.getlength(content)
            n_spaces = max(0, int(gap / space_w))
            result.append(full_space * n_spaces + content)
        else:
            result.append(line)

    return "\n".join(result)


def _split_manual_pages(text: str) -> List[str]:
    """按独占一行的 ``---`` 分割为手动分页段落。"""
    lines = text.split("\n")
    segments: List[str] = []
    current: List[str] = []

    for line in lines:
        if re.match(r"^\s*-{3,}\s*$", line):
            if current:
                segments.append("\n".join(current))
                current = []
        else:
            current.append(line)

    if current:
        segments.append("\n".join(current))

    return segments if segments else [""]


def _generate_pages(
    text: str,
    config: RenderConfig,
    layout: _Layout,
    font: ImageFont.FreeTypeFont,
    scale: int = 1,
) -> Iterable[Image.Image]:
    """调用 handrightbeta 生成手写页面，应用纸张背景与克制的不规则度/墨水参数。

    ``scale`` 为超采样倍数（默认 1）；``word_spacing`` 等未随 ``layout`` 自动
    缩放的像素参数在此显式乘上 scale。
    """
    background = _make_background(config, layout)
    ink_depth_sigma = config.ink_variation * layout.font_size
    fill = _parse_pen_color(config.pen_color)

    template = Template(
        background=background,
        font=font,
        fill=fill,
        left_margin=layout.margin_left,
        right_margin=layout.margin_right,
        top_margin=layout.margin_top,
        bottom_margin=layout.margin_bottom,
        line_spacing=layout.line_spacing,
        word_spacing=int(round(config.word_spacing * scale)),
        start_chars=_START_CHARS,
        end_chars=_END_CHARS,
        line_spacing_sigma=layout.font_size * config.line_spacing_ratio,
        font_size_sigma=layout.font_size * config.font_size_ratio,
        word_spacing_sigma=layout.font_size * config.word_spacing_ratio,
        perturb_x_sigma=layout.font_size * config.perturb_x_ratio,
        perturb_y_sigma=layout.font_size * config.perturb_y_ratio,
        perturb_theta_sigma=config.perturb_theta_sigma,
        ink_depth_sigma=ink_depth_sigma,
    )

    return handwrite(text, template, seed=config.seed)


def _make_background(config: RenderConfig, layout: _Layout) -> Image.Image:
    """生成引擎排版背景。

    照片背景不能直接交给文字引擎，否则文字会按整张照片坐标落在桌面上。
    ``background_id`` 存在时先返回规范化白纸，最终由
    :func:`_composite_page_on_photo_surface` 把墨迹映射到真实纸面。无照片时的
    程序化背景与旧行为逐字节一致。
    """
    if config.background_id:
        return Image.new(
            "RGB", (layout.page_width, layout.page_height), color=(255, 255, 255)
        )
    return _build_background_cached(
        config.paper_style,
        layout.page_width,
        layout.page_height,
        layout.margin_top,
        layout.margin_bottom,
        layout.margin_left,
        layout.margin_right,
        layout.font_size,
        layout.line_spacing,
    )


@functools.lru_cache(maxsize=16)
def _load_background_cached(
    background_id: str, page_width: int, page_height: int
) -> Image.Image:
    """开源版不包含真实背景图模块，直接返回白纸。"""
    return Image.new("RGB", (page_width, page_height), color=(255, 255, 255))


def _ink_density_layer(page: Image.Image, pen_color: str) -> Image.Image:
    """从白纸渲染页提取 0..255 墨迹覆盖密度，保留抗锯齿边缘。"""
    gray = np.asarray(page.convert("L"), dtype=np.float32)
    red, green, blue = _parse_pen_color(pen_color)
    pen_luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    denominator = max(32.0, 255.0 - pen_luminance)
    density = np.clip((255.0 - gray) / denominator, 0.0, 1.0)
    density[density < (1.0 / 255.0)] = 0.0
    return Image.fromarray(np.rint(density * 255.0).astype(np.uint8), mode="L")


def _composite_page_on_photo_surface(
    page: Image.Image, config: "RenderConfig"
) -> Image.Image:
    """开源版不包含照片纸面合成（需 paper_surface / backgrounds 闭源模块）。

    直接返回原页面。完整版见「手写如真」微信小程序。
    """
    return page.convert("RGB")


@functools.lru_cache(maxsize=32)
def _build_background_cached(
    paper_style: str,
    page_width: int,
    page_height: int,
    margin_top: int,
    margin_bottom: int,
    margin_left: int,
    margin_right: int,
    font_size: int,
    line_spacing: int,
) -> Image.Image:
    layout = _Layout(
        page_width=page_width,
        page_height=page_height,
        margin_top=margin_top,
        margin_bottom=margin_bottom,
        margin_left=margin_left,
        margin_right=margin_right,
        font_size=font_size,
        line_spacing=line_spacing,
    )
    if paper_style == "plain":
        return Image.new(
            "RGB", (layout.page_width, layout.page_height), color=(255, 255, 255)
        )
    if paper_style == "lined":
        return _lined_background(layout)
    if paper_style == "grid":
        return _grid_background(layout)
    raise ValueError(f"unsupported paper_style: {paper_style}")


def _lined_background(layout: _Layout) -> Image.Image:
    """横线纸：按行高画淡灰横线，与文本行基线对齐，不改动排版。"""
    img = Image.new("RGB", (layout.page_width, layout.page_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    x0, x1 = layout.margin_left, layout.page_width - layout.margin_right
    y = layout.margin_top + layout.line_spacing
    while y < layout.page_height - layout.margin_bottom:
        draw.line((x0, y, x1, y), fill=_LINED_COLOR, width=1)
        y += layout.line_spacing
    return img


def _grid_background(layout: _Layout) -> Image.Image:
    """方格纸：横线按行高、竖线按字宽画淡灰格线，不改动排版。"""
    img = Image.new("RGB", (layout.page_width, layout.page_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    x0, x1 = layout.margin_left, layout.page_width - layout.margin_right
    y0, y1 = layout.margin_top, layout.page_height - layout.margin_bottom

    y = layout.margin_top + layout.line_spacing
    while y < layout.page_height - layout.margin_bottom:
        draw.line((x0, y, x1, y), fill=_GRID_COLOR, width=1)
        y += layout.line_spacing

    x = layout.margin_left + layout.font_size
    while x < layout.page_width - layout.margin_right:
        draw.line((x, y0, x, y1), fill=_GRID_COLOR, width=1)
        x += layout.font_size

    return img


def _downscale_for_preview(image: Image.Image) -> Image.Image:
    """将完整渲染结果按 PREVIEW_SCALE 等比缩小，用于 preview 输出。"""
    return _downscale(image, PREVIEW_SCALE)


def _concat_pages(pages: Sequence[Image.Image]) -> Image.Image:
    """将多页垂直拼接为一张长图。"""
    max_width = max(p.width for p in pages)
    total_height = sum(p.height for p in pages)
    result = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))

    y = 0
    for page in pages:
        result.paste(page, (0, y))
        y += page.height

    return result


# PNG zlib 压缩级别。默认的 6 在整页上要 839ms，级别 1 只要 223ms，而文件
# 只大 15%（5.05 -> 5.79 MB）—— 成像链加过传感器噪声，这种图本来就压不动，
# 多花的 600ms 换不回多少字节。级别 0 不压缩，体积翻 3.7 倍，不可接受。
_PNG_COMPRESS_LEVEL = 1


def _image_to_bytes(
    image: Image.Image, label: Optional["aigc.AigcLabel"] = None
) -> bytes:
    """将 PIL Image 保存为 PNG 字节（可选写入 AIGC 隐式标识 tEXt）。"""
    buffer = io.BytesIO()
    image.save(
        buffer,
        format="PNG",
        pnginfo=aigc.png_metadata(label),
        compress_level=_PNG_COMPRESS_LEVEL,
    )
    return buffer.getvalue()


def _image_to_jpg(
    image: Image.Image,
    quality: int = 95,
    label: Optional["aigc.AigcLabel"] = None,
) -> bytes:
    """将 PIL Image 保存为 JPEG 字节（可选写入 AIGC 隐式标识 EXIF）。"""
    buffer = io.BytesIO()
    # handrightbeta 内部使用 "1" 或 "L" 模式，需转换为 RGB
    rgb_image = image.convert("RGB")
    exif = aigc.jpeg_exif(label)
    if exif is None:
        rgb_image.save(buffer, format="JPEG", quality=quality)
    else:
        rgb_image.save(buffer, format="JPEG", quality=quality, exif=exif)
    return buffer.getvalue()


def _images_to_pdf(
    images: Sequence[Image.Image],
    label: Optional["aigc.AigcLabel"] = None,
    photographic: bool = False,
    page_size_pt: Optional[Tuple[float, float]] = None,
) -> bytes:
    """将多页 PIL Image 保存为多页 PDF 字节（使用 ReportLab）。

    ``photographic``：页面是照片式的连续色调（v3 成像仿真 / 真实纸面照片）。
    这类页面用无损流压不动 —— 一页 A4 要 5MB，九页就是 46MB，手机根本下不
    下来。改成嵌入 JPEG 流（PDF 的 DCTDecode），体积降一个数量级，而且
    ReportLab 直接透传不再重新编码，顺带省掉整页 zlib 的时间。
    线稿式页面（旧引擎的纯黑字白底）反过来是无损流更小也更锐利，保持原路径。
    """
    if not images:
        raise ValueError("no images to convert to PDF")

    buffer = io.BytesIO()
    first = images[0]
    if page_size_pt is not None:
        page_width, page_height = page_size_pt
    else:
        page_width, page_height = first.width, first.height
    # invariant=1：固定 /CreationDate 等文档元数据，保证同输入逐字节可复现
    # （G5 稳定性要求：同 text+seed 的 PDF 跨运行 byte-identical）。
    c = canvas.Canvas(buffer, pagesize=(page_width, page_height), invariant=1)

    # AIGC 隐式标识写进 PDF 文档信息字典。这些值全部由内容确定性导出，
    # 不含时间戳，因此不破坏 invariant=1 的逐字节复现。
    info = aigc.pdf_info(label)
    if info:
        c.setProducer(info["producer"])
        c.setSubject(info["subject"])
        c.setKeywords(info["keywords"])

    for image in images:
        if page_size_pt is None and (
            image.width != page_width or image.height != page_height
        ):
            page_width, page_height = image.width, image.height
            c.setPageSize((page_width, page_height))

        img_buffer = io.BytesIO()
        if photographic:
            image.convert("RGB").save(
                img_buffer, format="JPEG", quality=_PDF_JPEG_QUALITY, subsampling=0
            )
        else:
            image.save(img_buffer, format="PNG", compress_level=_PNG_COMPRESS_LEVEL)
        img_buffer.seek(0)

        c.drawImage(
            ImageReader(img_buffer),
            0,
            0,
            width=page_width,
            height=page_height,
        )
        c.showPage()

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def _resolve_font_path() -> Path:
    """返回第一个可用的中文字体路径。

    优先系统字体（Windows 开发机 C:/Windows/Fonts）；若都不存在（如 Linux
    容器 / slim 镜像），回退到 manifest 中第一个 approved 开源字体（字体
    自包含在 assets/fonts，随镜像分发）。
    """
    for path in _FONT_CANDIDATES:
        p = Path(path)
        if p.exists():
            return p
    approved = fonts.load_manifest()
    if approved:
        return approved[0].file
    raise RuntimeError("No Chinese font found on system")


def _resolve_font_for(font_id: str = "") -> Path:
    """按 ``font_id`` 解析字体文件路径。

    ``font_id=""`` 或非法（不在 approved manifest）→ 回退系统字体，不崩溃。
    """
    if font_id:
        spec = fonts.get_approved_font(font_id)
        if spec is not None:
            return spec.file
    return _resolve_font_path()


@functools.lru_cache(maxsize=32)
def _load_font_file(path: str, size: int) -> ImageFont.FreeTypeFont:
    """按 (path, size) 缓存加载字体（font cache）。"""
    return ImageFont.truetype(path, size)


def _load_font_for(
    size: int, font_id: str = ""
) -> tuple[ImageFont.FreeTypeFont, Path]:
    """加载字体，返回 (字体对象, 字体文件路径)。"""
    path = _resolve_font_for(font_id)
    try:
        return _load_font_file(str(path), size), path
    except Exception:
        raise RuntimeError(f"Failed to load font: {path}")


def _load_font(size: int) -> tuple[ImageFont.FreeTypeFont, Path]:
    """加载系统字体，返回 (字体对象, 字体文件路径)。"""
    return _load_font_for(size)
