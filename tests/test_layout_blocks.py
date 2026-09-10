"""多块版式：IR 校验 + v3 按块落字，旧单文本路径不回归。"""

from io import BytesIO

from PIL import Image

from renderer.core import RenderConfig, render
from renderer.layout import (
    BBox,
    BlockStyle,
    LayoutBlock,
    LayoutDoc,
    LayoutPage,
    assign_columns,
    flatten_pages,
    make_table_from_cells,
    make_table_grid,
    specs_to_config_pages,
    tag_running_regions,
    validate_doc,
)


def test_validate_and_flatten_table() -> None:
    table = make_table_grid(BBox(x=10, y=40, w=80, h=20), 2, 2, prefix="t", texts=["甲", "乙", "丙", "丁"])
    doc = LayoutDoc(
        pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(id="h", type="heading", bbox=BBox(x=10, y=8, w=80, h=12), text="标题", style=BlockStyle(fontSize=32)),
                    table,
                ]
            )
        ]
    )
    assert validate_doc(doc, strict=True) == []
    pages = flatten_pages(doc)
    assert len(pages) == 1
    assert len(pages[0]) == 5  # heading + 4 cells


def test_v3_renders_two_blocks_and_table() -> None:
    table = make_table_grid(BBox(x=10, y=48, w=80, h=24), 2, 2, prefix="t", texts=["春", "夏", "秋", "冬"])
    doc = LayoutDoc(
        pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(id="a", type="paragraph", bbox=BBox(x=8, y=10, w=84, h=16), text="上块文字"),
                    LayoutBlock(id="b", type="paragraph", bbox=BBox(x=8, y=28, w=84, h=16), text="下块文字"),
                    table,
                ]
            )
        ]
    )
    layout_pages = specs_to_config_pages(flatten_pages(doc))
    data = render(
        "上块文字\n下块文字\n春夏秋冬",
        config=RenderConfig(
            engine="v3",
            mode="preview",
            max_pages=1,
            layout_pages=layout_pages,
            text_align="left",
        ),
    )
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    image = Image.open(BytesIO(data))
    assert image.size[0] > 0


def test_header_footer_skip_overlap_and_flatten_style() -> None:
    header = LayoutBlock(id="hd", type="paragraph", bbox=BBox(x=8, y=1, w=84, h=6), text="页眉")
    footer = LayoutBlock(id="ft", type="paragraph", bbox=BBox(x=8, y=93, w=84, h=5), text="页脚")
    body = LayoutBlock(id="bd", type="paragraph", bbox=BBox(x=8, y=4, w=84, h=20), text="正文")
    blocks = [header, footer, body]
    tag_running_regions(blocks)
    assert header.type == "header"
    assert footer.type == "footer"
    doc = LayoutDoc(pages=[LayoutPage(blocks=blocks)])
    assert validate_doc(doc, strict=True) == []
    specs = flatten_pages(doc)[0]
    assert any(spec.style_id == "" and spec.glyph_scale <= 0.85 for spec in specs if spec.text == "页眉")


def test_assign_columns_two_sides() -> None:
    blocks = [
        LayoutBlock(id="l0", type="paragraph", bbox=BBox(x=6, y=12, w=36, h=10), text="左一"),
        LayoutBlock(id="l1", type="paragraph", bbox=BBox(x=6, y=28, w=36, h=10), text="左二"),
        LayoutBlock(id="r0", type="paragraph", bbox=BBox(x=58, y=12, w=36, h=10), text="右一"),
        LayoutBlock(id="r1", type="paragraph", bbox=BBox(x=58, y=28, w=36, h=10), text="右二"),
    ]
    assert assign_columns(blocks) is True
    assert [b.column for b in blocks] == [0, 0, 1, 1]


def test_merged_cell_keeps_union_bbox() -> None:
    cells = [
        LayoutBlock(id="c0", type="cell", bbox=BBox(x=10, y=40, w=80, h=10), text="合并", colspan=2),
        LayoutBlock(id="c1", type="cell", bbox=BBox(x=10, y=50, w=40, h=10), text="左"),
        LayoutBlock(id="c2", type="cell", bbox=BBox(x=50, y=50, w=40, h=10), text="右"),
    ]
    table = make_table_from_cells(BBox(x=10, y=40, w=80, h=20), cells, prefix="t")
    pages = flatten_pages(LayoutDoc(pages=[LayoutPage(blocks=[table])]))
    assert len(pages[0]) == 3
    merged = next(spec for spec in pages[0] if spec.text == "合并")
    assert merged.w == 80


def test_v3_renders_two_block_styles() -> None:
    doc = LayoutDoc(
        pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(
                        id="a",
                        type="paragraph",
                        bbox=BBox(x=8, y=10, w=84, h=18),
                        text="楷书块",
                        style=BlockStyle(styleId="kai-neat"),
                    ),
                    LayoutBlock(
                        id="b",
                        type="paragraph",
                        bbox=BBox(x=8, y=36, w=84, h=18),
                        text="行书块",
                        style=BlockStyle(styleId="xingshu-fast"),
                    ),
                ]
            )
        ]
    )
    pages = flatten_pages(doc)
    assert pages[0][0].style_id == "kai-neat"
    assert pages[0][1].style_id == "xingshu-fast"
    data = render(
        "楷书块\n行书块",
        config=RenderConfig(
            engine="v3",
            mode="preview",
            max_pages=1,
            layout_pages=specs_to_config_pages(pages),
        ),
    )
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_plain_text_path_still_renders() -> None:
    data = render(
        "兼容旧路径",
        config=RenderConfig(engine="v3", mode="preview", max_pages=1, text_align="center"),
    )
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
