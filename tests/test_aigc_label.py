"""AIGC 生成合成内容标识（《标识办法》/ GB 45438-2025）。

两条硬约束：
1. 产物必须同时带隐式标识（文件元数据）和显式标识（页面可见提示）。
2. 加标识不能破坏渲染器的逐字节可复现性（见 r2/g5/test_reproducibility.py），
   所以内容编号只能由输入哈希导出，不能用时间戳或随机数。
"""

import json
from io import BytesIO

import pytest
from PIL import Image, ImageChops

from renderer import RenderConfig, aigc, render, render_with_metadata

TEXT = "标识测试用文本，中文与 English 混排。"
PRODUCER = "handwrite-real-test"


def _cfg(**kwargs) -> RenderConfig:
    base = dict(
        mode="preview",
        preview_width=360,
        engine="handright",
        seed=4242,
        aigc_producer=PRODUCER,
    )
    base.update(kwargs)
    return RenderConfig(**base)


# --------------------------------------------------------------- 隐式标识


def test_png_carries_implicit_label():
    data, meta = render_with_metadata(TEXT, _cfg())
    image = Image.open(BytesIO(data))
    image.load()
    payload = json.loads(image.info[aigc.METADATA_KEY])
    assert payload["Label"] == aigc.LABEL_SYNTHETIC
    assert payload["ContentProducer"] == PRODUCER
    assert payload["ProduceID"]
    assert meta["aigc"]["produce_id"] == payload["ProduceID"]


def test_jpeg_carries_implicit_label_in_exif():
    data, _meta = render_with_metadata(TEXT, _cfg(format="jpg"))
    image = Image.open(BytesIO(data))
    image.load()
    exif = image.getexif()
    description = str(exif.get(0x010E))
    assert description.startswith(f"{aigc.METADATA_KEY}=")
    payload = json.loads(description.split("=", 1)[1])
    assert payload["ContentProducer"] == PRODUCER
    assert exif.get(0x0131) == PRODUCER


def test_pdf_carries_implicit_label_in_document_info():
    data, _meta = render_with_metadata(TEXT, _cfg(format="pdf"))
    assert aigc.METADATA_KEY.encode() in data
    assert PRODUCER.encode() in data


def test_library_default_writes_no_label():
    """渲染器作为库被直接调用时不冒充服务提供者。"""
    data, meta = render_with_metadata(TEXT, RenderConfig(mode="preview", preview_width=360))
    image = Image.open(BytesIO(data))
    image.load()
    assert aigc.METADATA_KEY not in image.info
    assert "aigc" not in meta


# --------------------------------------------------------------- 显式标识


def test_visible_label_changes_pixels():
    with_label = Image.open(BytesIO(render(TEXT, _cfg(aigc_visible=True)))).convert("RGB")
    without = Image.open(BytesIO(render(TEXT, _cfg(aigc_visible=False)))).convert("RGB")
    assert with_label.size == without.size
    difference = ImageChops.difference(with_label, without).getbbox()
    assert difference is not None, "显式标识必须在页面上留下可见痕迹"
    # 标识落在右下角：差异区域应当靠近页面右下。
    left, top, right, bottom = difference
    assert right > with_label.width * 0.7
    assert bottom > with_label.height * 0.85


def test_visible_label_is_small_enough_to_keep_page_usable():
    """显式标识要显著但不能盖住正文，占比控制在页面 5% 以内。"""
    with_label = Image.open(BytesIO(render(TEXT, _cfg(aigc_visible=True)))).convert("RGB")
    without = Image.open(BytesIO(render(TEXT, _cfg(aigc_visible=False)))).convert("RGB")
    left, top, right, bottom = ImageChops.difference(with_label, without).getbbox()
    area_ratio = ((right - left) * (bottom - top)) / (with_label.width * with_label.height)
    assert area_ratio < 0.05


def test_implicit_label_survives_disabling_visible_label():
    """关掉显式标识时隐式标识仍在（生产禁止关，但两者互相独立）。"""
    data, _meta = render_with_metadata(TEXT, _cfg(aigc_visible=False))
    image = Image.open(BytesIO(data))
    image.load()
    assert aigc.METADATA_KEY in image.info


# --------------------------------------------------------------- 可复现性


@pytest.mark.parametrize("fmt", ["png", "jpg", "pdf"])
def test_labeled_output_stays_byte_identical(fmt: str):
    cfg = _cfg(format=fmt)
    assert render(TEXT, cfg) == render(TEXT, cfg)


def test_produce_id_is_content_derived_not_random():
    same = aigc.build_label(TEXT, _cfg(), PRODUCER)
    again = aigc.build_label(TEXT, _cfg(), PRODUCER)
    other = aigc.build_label(TEXT + "。", _cfg(), PRODUCER)
    assert same.produce_id == again.produce_id
    assert same.produce_id != other.produce_id


def test_produce_id_changes_with_render_parameters():
    """同一段文字换了笔迹或纸张就是另一份产物，编号必须不同。"""
    base = aigc.build_label(TEXT, _cfg(), PRODUCER)
    other_seed = aigc.build_label(TEXT, _cfg(seed=9999), PRODUCER)
    assert base.produce_id != other_seed.produce_id


def test_metadata_payload_follows_national_standard_fields():
    label = aigc.build_label(TEXT, _cfg(), PRODUCER)
    payload = json.loads(label.to_metadata_json())
    assert set(payload) == {
        "Label",
        "ContentProducer",
        "ProduceID",
        "ReservedCode1",
        "ContentPropagator",
        "PropagateID",
    }
