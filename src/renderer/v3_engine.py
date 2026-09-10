"""V3「真迹」引擎的适配层。

``core.RenderConfig`` 用的是像素逻辑坐标，V3 管线用的是毫米 + DPI 的物理坐标 ——
因为墨迹扩散、纸张纤维、笔尖足迹都只有在物理尺度下才有确定的量纲。这一层负责
两套坐标系的换算，并把产品侧的枚举（paper_style / filter_style / background_id）
翻译成物理参数。

约束：V3 与其它引擎共用 ``core.render()`` 的后处理（滤镜 / preview 缩放 /
照片合成 / 格式输出），因此这里只返回逻辑分辨率的 RGB 页面。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

from PIL import Image

if TYPE_CHECKING:  # pragma: no cover
    from . import core

# preset 的物理纸张尺寸（毫米）。core 的像素 preset 必须与它保持同一比例。
_PRESET_MM: Dict[str, Tuple[float, float]] = {
    "a4": (210.0, 297.0),
    "notebook": (176.0, 248.0),
    "letter": (215.9, 279.4),
}

# filter_style 决定成像链强度：产品语义上 none 是"扫描件"、scan 是"手机翻拍"。
#
# 传感器噪声按 g8 笔记本语料标定：真人照片纸面区的高频残差尺度中位 0.0047，
# 此前 soft 只能做到 0.0021 —— 纸面比真实照片干净一半，是合成检测最直接的
# 抓手。三档等比放大到实测量级；扫描件本来就更干净，所以 none 仍最低。
_CAPTURE_PRESETS = {
    "none": dict(
        illumination_strength=0.04,
        vignette=0.03,
        defocus_px=0.30,
        chromatic_px=0.06,
        shot_noise=0.0105,
        read_noise=0.0045,
        sharpen=0.18,
        jpeg_quality=95,
        curl=0.001,
    ),
    "soft": dict(
        illumination_strength=0.09,
        vignette=0.07,
        defocus_px=0.48,
        chromatic_px=0.14,
        shot_noise=0.0165,
        read_noise=0.0072,
        sharpen=0.22,
        jpeg_quality=92,
        curl=0.003,
    ),
    "scan": dict(
        illumination_strength=0.14,
        vignette=0.11,
        defocus_px=0.58,
        chromatic_px=0.24,
        shot_noise=0.0240,
        read_noise=0.0096,
        sharpen=0.30,
        jpeg_quality=88,
        curl=0.005,
    ),
}


class V3Engine:
    """``engine="v3"``：骨架轨迹 + 墨迹物理 + 成像仿真。"""

    def render(self, text: str, config: "core.RenderConfig"):
        from . import core as core_mod
        from .engine import RenderResult
        from .v3.paper import CaptureSpec, PaperSpec
        from .v3.pipeline import PageSpec, RenderRequest
        from .v3.pipeline import render as v3_render
        from .v3.style import get_style

        layout = core_mod._resolve_layout(config)
        width_mm, height_mm = _PRESET_MM.get(config.preset, _PRESET_MM["a4"])
        layout_ppm = layout.page_width / width_mm
        render_scale = 1.0
        # Preview renders at the output pixel size. Layout stays in millimetres,
        # so wrapping and pagination match HD; only the raster is cheaper.
        if config.mode == "preview":
            from .core import PREVIEW_SCALE

            target_w = config.preview_width or max(1, int(round(layout.page_width * PREVIEW_SCALE)))
            render_scale = target_w / max(layout.page_width, 1)
            output_size = (
                target_w,
                max(1, int(round(layout.page_height * render_scale))),
            )
        else:
            output_size = (layout.page_width, layout.page_height)
        px_per_mm = layout_ppm * render_scale
        dpi = px_per_mm * 25.4

        page = PageSpec(
            dpi=dpi,
            width_mm=width_mm,
            height_mm=height_mm,
            margin_left_mm=layout.margin_left / layout_ppm,
            margin_right_mm=layout.margin_right / layout_ppm,
            margin_top_mm=layout.margin_top / layout_ppm,
            margin_bottom_mm=layout.margin_bottom / layout_ppm,
            glyph_size_mm=layout.font_size / layout_ppm,
            line_pitch_mm=layout.line_spacing / layout_ppm,
            tracking=config.word_spacing / max(layout.font_size, 1),
            indent=config.paragraph_indent,
        )

        if config.background_id:
            pass  # 开源版不包含横线检测与照片纸面合成，背景参数忽略

        paper = _resolve_paper(config, page)
        capture = _resolve_capture(config)
        style = get_style(_resolve_style_id(config.stroke_writer_id))
        slant = float(getattr(config, "text_slant", 0.0) or 0.0)
        if slant:
            from dataclasses import replace

            style = replace(
                style,
                structure=replace(style.structure, slant=style.structure.slant + slant),
            )
        resolved_pen = _resolve_pen(style, config)

        result = v3_render(
            RenderRequest(
                text=text,
                style_id=style.style_id,
                page=page,
                paper=paper,
                capture=capture,
                seed=config.seed,
                supersample=_supersample_for(config),
                max_pages=config.max_pages,
                layout_pages=_layout_pages_from_config(config),
                text_align=getattr(config, "text_align", "left") or "left",
                pen=None if resolved_pen is style.pen else resolved_pen,
                scribble_count=int(getattr(config, "scribble_count", 0) or 0),
                scribble_style=str(getattr(config, "scribble_style", None) or "strike"),
            )
        )

        # 缺字必须报错而不是安静跳过：v3 没有"豆腐块"可画，跳过的字会在
        # 成品里凭空消失，用户直到打印出来才发现。与 handright 的缺字检测
        # 保持同一契约（ValueError -> API 400）。
        missing = result.metadata.get("missing_glyphs") or []
        if missing:
            from . import core as core_mod

            raise ValueError(core_mod._missing_message(missing))

        pages: List[Image.Image] = [Image.fromarray(image) for image in result.images]
        pages = [
            page_image
            if page_image.size == output_size
            else page_image.resize(output_size, Image.Resampling.LANCZOS)
            for page_image in pages
        ]
        return RenderResult(
            pages=pages, total_pages=int(result.metadata.get("total_pages") or len(pages))
        )


def _layout_pages_from_config(config: "core.RenderConfig"):
    raw = getattr(config, "layout_pages", None)
    if not raw:
        return None
    from .layout import BlockSpec

    pages = []
    for page in raw:
        specs = []
        for item in page or ():
            if not isinstance(item, dict):
                continue
            align = str(item.get("align") or "left")
            if align not in ("left", "center", "right"):
                align = "left"
            specs.append(
                BlockSpec(
                    text=str(item.get("text") or ""),
                    x=float(item.get("x") or 0),
                    y=float(item.get("y") or 0),
                    w=float(item.get("w") or 100),
                    h=float(item.get("h") or 100),
                    align=align,
                    indent=int(item.get("indent") or 0),
                    glyph_scale=float(item.get("glyph_scale") or 1.0),
                    erase=bool(item.get("erase")),
                    style_id=str(item.get("style_id") or item.get("styleId") or ""),
                )
            )
        pages.append(specs)
    return pages or None


# 日常行楷默认蓝黑。换色时若仍是这管墨，吸收系数不要改。
_DEFAULT_INK_RGB = (26, 32, 58)
_DEFAULT_ABSORB_RGB = (1.05, 1.0, 0.86)
_UNSET_PEN_COLOR = "#232323"


def _absorb_for_ink(
    rgb: tuple[int, int, int], fallback: tuple[float, float, float]
) -> tuple[float, float, float]:
    if rgb == _DEFAULT_INK_RGB:
        return _DEFAULT_ABSORB_RGB
    red, green, blue = rgb
    if red > blue + 30 and red > green + 20:
        return (0.82, 1.12, 1.18)
    if blue > red + 20:
        return (1.14, 1.04, 0.72)
    return fallback


def _resolve_pen(style, config):
    """把 RenderConfig 的 pen_id / pen_color 落到一支 Pen。无改动时返回 style.pen。"""
    from dataclasses import replace

    from .core import _parse_pen_color
    from .v3.style import PENS

    pen = style.pen
    pen_id = getattr(config, "pen_id", "") or ""
    if pen_id:
        if pen_id not in PENS:
            raise ValueError(f"unsupported pen_id: {pen_id}")
        if pen_id != pen.pen_id:
            pen = PENS[pen_id]
    color = str(getattr(config, "pen_color", "") or "")
    if color and color.lower() != _UNSET_PEN_COLOR:
        rgb = _parse_pen_color(color)
        if rgb != pen.ink_rgb:
            pen = replace(pen, ink_rgb=rgb, absorb_rgb=_absorb_for_ink(rgb, pen.absorb_rgb))
    return pen


def _resolve_style_id(writer_id: str) -> str:
    from .v3.style import DEFAULT_STYLE_ID, has_style

    if not writer_id:
        return DEFAULT_STYLE_ID
    if has_style(writer_id):
        return writer_id
    if writer_id.startswith('fit-'):
        raise ValueError(f'unknown personal style: {writer_id}')
    return DEFAULT_STYLE_ID


def _resolve_paper(config: "core.RenderConfig", page) -> "object":
    from .v3.paper import PaperSpec

    # 套打与照片背景都出规范化白纸：套打的格在用户那张实体纸上，不能再画
    # 一层程序格；照片背景的纸纹来自真实照片，这里再铺会变成两层纸。
    if getattr(config, "output_mode", "page") == "plate" or config.background_id:
        return PaperSpec(
            paper_id="normalized-white",
            base_rgb=(255, 255, 255),
            fiber_amplitude=0.0,
            blotch_amplitude=0.0,
            ruling="blank",
        )

    if config.paper_style == "lined":
        return PaperSpec(
            paper_id="ruled-notebook",
            ruling="lined",
            line_pitch_mm=page.resolved_line_pitch_mm,
            margin_rule_mm=page.margin_left_mm * 0.72,
        )
    if config.paper_style == "grid":
        return PaperSpec(
            paper_id="grid-notebook",
            ruling="grid",
            line_pitch_mm=page.resolved_line_pitch_mm,
        )
    return PaperSpec()


def _resolve_capture(config: "core.RenderConfig") -> "object":
    from .v3.paper import CaptureSpec

    # 套打要干净墨迹；照片背景的成像统计由真实照片提供，再叠一层会双重劣化。
    if getattr(config, "output_mode", "page") == "plate" or config.background_id:
        return CaptureSpec(enabled=False)
    if config.mode == "preview":
        # preview 会被缩到一半，成像细节看不见，省掉这段耗时
        return CaptureSpec(enabled=False)
    return CaptureSpec(**_CAPTURE_PRESETS.get(config.filter_style, _CAPTURE_PRESETS["soft"]))


def _supersample_for(config: "core.RenderConfig") -> int:
    # 254 DPI 上 2× 超采样已经等价 ~500 DPI 的墨迹场；再高主要是算力。
    if config.mode == "preview":
        return 1
    return {"off": 2, "standard": 2, "high": 3}.get(config.antialias, 2)
