"""AIGC 生成合成内容标识（《人工智能生成合成内容标识办法》/ GB 45438-2025）。

办法要求生成合成内容同时带两种标识：

- **隐式标识**：写进文件元数据，含内容属性、服务提供者编码、内容编号。
  本模块按 GB 45438-2025 的字段结构写一个名为 ``AIGC`` 的 JSON 元数据段，
  PNG 落 tEXt、JPEG 落 EXIF、PDF 落文档信息字典。
- **显式标识**：在图片上可感知的位置画提示文字。

内容编号必须由内容本身哈希导出而不能用时间戳或随机数：渲染器有跨进程
逐字节复现的测试（``tests/r2/g5/test_reproducibility.py``），任何非确定性
字段都会让同一 seed 产出不同字节。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

if TYPE_CHECKING:  # pragma: no cover
    from . import core

# 元数据段名。GB 45438-2025 规定隐式标识写在文件元数据的 "AIGC" 字段里。
METADATA_KEY = "AIGC"

# Label=1 表示"生成合成内容"。
LABEL_SYNTHETIC = "1"

# EXIF 标签号：0x010E ImageDescription / 0x0131 Software。
_EXIF_IMAGE_DESCRIPTION = 0x010E
_EXIF_SOFTWARE = 0x0131

# 显式标识默认文案。中英并列，避免只有中文时海外端不可读。
DEFAULT_VISIBLE_TEXT = "AI生成"


@dataclass(frozen=True)
class AigcLabel:
    """一次渲染对应的标识信息。全部字段可由输入确定性导出。"""

    producer: str
    produce_id: str
    visible_text: str = DEFAULT_VISIBLE_TEXT

    def to_metadata_json(self) -> str:
        """GB 45438-2025 隐式标识字段结构。字段顺序固定，保证字节可复现。"""
        payload = {
            "Label": LABEL_SYNTHETIC,
            "ContentProducer": self.producer,
            "ProduceID": self.produce_id,
            "ReservedCode1": "",
            "ContentPropagator": "",
            "PropagateID": "",
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def to_dict(self) -> Dict[str, str]:
        return {
            "label": LABEL_SYNTHETIC,
            "producer": self.producer,
            "produce_id": self.produce_id,
        }


def build_label(
    text: str, config: "core.RenderConfig", producer: str
) -> AigcLabel:
    """由正文与渲染配置确定性导出标识。

    ``produce_id`` 是内容指纹而不是用户标识：它只包含正文与渲染参数的哈希，
    不含 openid、请求 id 或时间，因此既能溯源同一份产物，又不会把用户身份
    写进用户可以自由传播的文件里。
    """
    fingerprint = "\u0000".join(
        (
            producer,
            config.engine,
            str(config.seed),
            config.stroke_writer_id or "",
            config.font_id or "",
            config.background_id or "",
            config.preset,
            config.format,
            config.mode,
            text,
        )
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:32]
    return AigcLabel(producer=producer, produce_id=digest)


# ---------------------------------------------------------------------------
# 隐式标识：文件元数据
# ---------------------------------------------------------------------------


def png_metadata(label: Optional[AigcLabel]) -> Optional[PngImagePlugin.PngInfo]:
    """PNG tEXt 块。``None`` 表示不写标识（调用方已显式关闭）。"""
    if label is None:
        return None
    info = PngImagePlugin.PngInfo()
    info.add_text(METADATA_KEY, label.to_metadata_json())
    # Software 是通用看图软件也能显示的字段，作为人眼可读的冗余。
    info.add_text("Software", label.producer)
    return info


def jpeg_exif(label: Optional[AigcLabel]) -> Optional[bytes]:
    """JPEG EXIF 字节。写 ImageDescription + Software 两个通用可读字段。"""
    if label is None:
        return None
    exif = Image.Exif()
    exif[_EXIF_IMAGE_DESCRIPTION] = f"{METADATA_KEY}={label.to_metadata_json()}"
    exif[_EXIF_SOFTWARE] = label.producer
    return exif.tobytes()


def pdf_info(label: Optional[AigcLabel]) -> Dict[str, str]:
    """PDF 文档信息字典条目（供 reportlab canvas 设置）。"""
    if label is None:
        return {}
    return {
        "producer": label.producer,
        "subject": f"{METADATA_KEY}={label.to_metadata_json()}",
        "keywords": f"{METADATA_KEY} Label={LABEL_SYNTHETIC} "
        f"ContentProducer={label.producer} ProduceID={label.produce_id}",
    }


# ---------------------------------------------------------------------------
# 显式标识：图片上的可见提示
# ---------------------------------------------------------------------------


def draw_visible_label(image: Image.Image, label: Optional[AigcLabel]) -> Image.Image:
    """在页面右下角画提示文字，返回新图（不改动入参）。

    位置与字号按页面尺寸等比，保证 preview 与 hd 上的观感一致；颜色用中灰
    而不是纯黑，避免被误认成正文墨迹。
    """
    if label is None or not label.visible_text:
        return image

    canvas_image = image.convert("RGB") if image.mode != "RGB" else image.copy()
    width, height = canvas_image.size
    if min(width, height) < 64:
        return canvas_image

    font_px = max(12, round(min(width, height) * 0.016))
    font = _load_label_font(font_px)
    draw = ImageDraw.Draw(canvas_image)

    text = label.visible_text
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = right - left, bottom - top
    pad = max(4, round(font_px * 0.35))
    margin = max(8, round(min(width, height) * 0.012))

    box_right = width - margin
    box_bottom = height - margin
    box_left = box_right - text_w - pad * 2
    box_top = box_bottom - text_h - pad * 2

    # 半透明白底：纸面深浅不一时保证提示始终可读。
    plate = Image.new("RGBA", (box_right - box_left, box_bottom - box_top), (255, 255, 255, 205))
    canvas_image.paste(plate, (box_left, box_top), plate)

    draw.text(
        (box_left + pad - left, box_top + pad - top),
        text,
        font=font,
        fill=(108, 108, 112),
    )
    return canvas_image


def _load_label_font(size_px: int) -> ImageFont.ImageFont:
    """标识文字用的字体。取不到中文字体时退回默认位图字体，绝不抛错。"""
    from . import core as core_mod

    try:
        return ImageFont.truetype(str(core_mod._resolve_font_path()), size_px)
    except Exception:  # noqa: BLE001 - 标识不能因为字体问题阻断渲染
        return ImageFont.load_default()


def label_size_hint(image_size: Tuple[int, int]) -> int:
    """显式标识占用的右下角高度（像素），供排版避让使用。"""
    width, height = image_size
    font_px = max(12, round(min(width, height) * 0.016))
    return font_px * 3
