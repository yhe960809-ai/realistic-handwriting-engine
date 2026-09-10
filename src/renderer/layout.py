"""版式中间层：一页纸上的多块文字。

抄写、导入文档、多区域编辑、原位换字都读写同一份 IR。
坐标用页百分比（0–100），与小程序区域框一致；毫米/像素只在渲染器内部换算。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal

BLOCK_TYPES = ("paragraph", "heading", "table", "cell", "box", "header", "footer", "column")
ALIGN_VALUES = ("left", "center", "right")
MAX_PAGES = 50
MAX_BLOCKS_PER_PAGE = 60
MAX_TEXT_LENGTH = 20_000
MIN_BOX_W = 4.0
MIN_BOX_H = 3.0


class LayoutError(ValueError):
    """版式不合法，应对调用方报错而不是静默截断。"""


@dataclass
class BBox:
    x: float = 0.0
    y: float = 0.0
    w: float = 100.0
    h: float = 100.0

    def clamped(self) -> BBox:
        x = max(0.0, min(100.0, float(self.x)))
        y = max(0.0, min(100.0, float(self.y)))
        w = max(0.0, min(100.0 - x, float(self.w)))
        h = max(0.0, min(100.0 - y, float(self.h)))
        return BBox(x=x, y=y, w=w, h=h)

    def overlaps(self, other: BBox, slack: float = 0.8) -> bool:
        a, b = self.clamped(), other.clamped()
        return not (
            a.x + a.w <= b.x + slack
            or b.x + b.w <= a.x + slack
            or a.y + a.h <= b.y + slack
            or b.y + b.h <= a.y + slack
        )


@dataclass
class BlockStyle:
    fontSize: int | None = None
    align: str = "left"
    indent: int = 0
    styleId: str = ""


@dataclass
class LayoutBlock:
    id: str
    type: str = "paragraph"
    bbox: BBox = field(default_factory=BBox)
    text: str = ""
    style: BlockStyle = field(default_factory=BlockStyle)
    erase: bool = False
    cells: list[LayoutBlock] = field(default_factory=list)
    rowspan: int = 1
    colspan: int = 1
    script: str = ""
    column: int | None = None


@dataclass
class LayoutPage:
    width_mm: float = 210.0
    height_mm: float = 297.0
    background: dict[str, Any] | None = None
    blocks: list[LayoutBlock] = field(default_factory=list)


@dataclass
class LayoutDoc:
    source: dict[str, Any] = field(default_factory=dict)
    pages: list[LayoutPage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def bbox_from_dict(raw: Any) -> BBox:
    data = raw if isinstance(raw, dict) else {}
    return BBox(
        x=_as_float(data.get("x"), 0.0),
        y=_as_float(data.get("y"), 0.0),
        w=_as_float(data.get("w"), 100.0),
        h=_as_float(data.get("h"), 100.0),
    ).clamped()


def block_from_dict(raw: Any, fallback_id: str) -> LayoutBlock:
    data = raw if isinstance(raw, dict) else {}
    block_id = str(data.get("id") or fallback_id)
    block_type = str(data.get("type") or "paragraph")
    if block_type not in BLOCK_TYPES:
        block_type = "box"
    style_raw = data.get("style") if isinstance(data.get("style"), dict) else {}
    align = str(style_raw.get("align") or data.get("align") or "left")
    if align not in ALIGN_VALUES:
        align = "left"
    indent = int(style_raw.get("indent") or data.get("indent") or 0)
    font_size = style_raw.get("fontSize", data.get("fontSize"))
    try:
        font_size_i = int(font_size) if font_size is not None else None
    except (TypeError, ValueError):
        font_size_i = None
    style_id = str(style_raw.get("styleId") or data.get("styleId") or data.get("font") or "")
    cells_raw = data.get("cells") if isinstance(data.get("cells"), list) else []
    cells = [
        block_from_dict(cell, f"{block_id}-c{index}")
        for index, cell in enumerate(cells_raw)
    ]
    try:
        rowspan = max(1, int(data.get("rowspan") or 1))
        colspan = max(1, int(data.get("colspan") or 1))
    except (TypeError, ValueError):
        rowspan, colspan = 1, 1
    column = data.get("column")
    try:
        column_i = int(column) if column is not None else None
    except (TypeError, ValueError):
        column_i = None
    return LayoutBlock(
        id=block_id,
        type=block_type,
        bbox=bbox_from_dict(data.get("bbox") or data),
        text=str(data.get("text") or data.get("content") or ""),
        style=BlockStyle(
            fontSize=font_size_i,
            align=align,
            indent=max(0, min(8, indent)),
            styleId=style_id,
        ),
        erase=bool(data.get("erase")),
        cells=cells,
        rowspan=rowspan,
        colspan=colspan,
        script=str(data.get("script") or ""),
        column=column_i,
    )


def page_from_dict(raw: Any) -> LayoutPage:
    data = raw if isinstance(raw, dict) else {}
    blocks_raw = data.get("blocks") if isinstance(data.get("blocks"), list) else []
    return LayoutPage(
        width_mm=max(50.0, min(500.0, _as_float(data.get("width_mm"), 210.0))),
        height_mm=max(50.0, min(700.0, _as_float(data.get("height_mm"), 297.0))),
        background=data.get("background") if isinstance(data.get("background"), dict) else None,
        blocks=[block_from_dict(item, f"b{index}") for index, item in enumerate(blocks_raw)],
    )


def doc_from_dict(raw: Any) -> LayoutDoc:
    data = raw if isinstance(raw, dict) else {}
    pages_raw = data.get("pages") if isinstance(data.get("pages"), list) else []
    warnings = [str(item) for item in (data.get("warnings") or []) if item]
    source = data.get("source") if isinstance(data.get("source"), dict) else {}
    return LayoutDoc(
        source=source,
        pages=[page_from_dict(page) for page in pages_raw],
        warnings=warnings,
    )


def _block_to_dict(block: LayoutBlock) -> dict[str, Any]:
    payload = {
        "id": block.id,
        "type": block.type,
        "bbox": asdict(block.bbox),
        "text": block.text,
        "style": asdict(block.style),
        "erase": block.erase,
        "rowspan": block.rowspan,
        "colspan": block.colspan,
        "script": block.script,
        "column": block.column,
    }
    if block.cells:
        payload["cells"] = [_block_to_dict(cell) for cell in block.cells]
    return payload


def doc_to_dict(doc: LayoutDoc) -> dict[str, Any]:
    return {
        "source": doc.source,
        "warnings": list(doc.warnings),
        "pages": [
            {
                "width_mm": page.width_mm,
                "height_mm": page.height_mm,
                "background": page.background,
                "blocks": [_block_to_dict(block) for block in page.blocks],
            }
            for page in doc.pages
        ],
    }


def iter_leaf_blocks(blocks: Iterable[LayoutBlock]) -> Iterable[LayoutBlock]:
    for block in blocks:
        if block.type == "table" and block.cells:
            yield from iter_leaf_blocks(block.cells)
        else:
            yield block


def joined_text(doc: LayoutDoc) -> str:
    parts: list[str] = []
    for page in doc.pages:
        for block in iter_leaf_blocks(page.blocks):
            if block.text:
                parts.append(block.text)
    return "\n".join(parts)


def validate_doc(doc: LayoutDoc, *, strict: bool = True) -> list[str]:
    """校验版式。strict 时超限抛 LayoutError；否则只收集 warning。"""
    warnings: list[str] = []
    if not doc.pages:
        warnings.append("文档没有可渲染的页")
    if len(doc.pages) > MAX_PAGES:
        message = f"页数超过 {MAX_PAGES} 页上限"
        if strict:
            raise LayoutError(message)
        warnings.append(message)
    text_len = len(joined_text(doc))
    if text_len > MAX_TEXT_LENGTH:
        message = f"正文超过 {MAX_TEXT_LENGTH} 字上限"
        if strict:
            raise LayoutError(message)
        warnings.append(message)
    for page_i, page in enumerate(doc.pages):
        leaves = list(iter_leaf_blocks(page.blocks))
        if len(leaves) > MAX_BLOCKS_PER_PAGE:
            message = f"第 {page_i + 1} 页块数超过 {MAX_BLOCKS_PER_PAGE}"
            if strict:
                raise LayoutError(message)
            warnings.append(message)
        for block in leaves:
            box = block.bbox.clamped()
            if box.w < MIN_BOX_W or box.h < MIN_BOX_H:
                warnings.append(f"块 {block.id} 过小，渲染时可能被跳过")
        running = {"header", "footer"}
        for i, left in enumerate(leaves):
            for right in leaves[i + 1 :]:
                if left.type in running or right.type in running:
                    continue
                if left.bbox.overlaps(right.bbox):
                    warnings.append(f"块 {left.id} 与 {right.id} 重叠")
    return warnings


def from_regions(
    regions: list[Any] | None,
    *,
    width_mm: float = 210.0,
    height_mm: float = 297.0,
    background: dict[str, Any] | None = None,
) -> LayoutDoc | None:
    """把小程序 regions 列表收成单页 LayoutDoc。空列表返回 None。"""
    if not regions:
        return None
    blocks: list[LayoutBlock] = []
    for index, raw in enumerate(regions):
        if not isinstance(raw, dict):
            continue
        block = block_from_dict(raw, f"r{index}")
        if block.type == "table" and not block.cells and raw.get("rows") and raw.get("cols"):
            block.cells = _grid_cells(
                block.bbox,
                int(raw["rows"]),
                int(raw["cols"]),
                prefix=block.id,
                texts=raw.get("cellTexts") or [],
            )
            block.type = "table"
        blocks.append(block)
    if not blocks:
        return None
    return LayoutDoc(
        source={"kind": "regions"},
        pages=[LayoutPage(width_mm=width_mm, height_mm=height_mm, background=background, blocks=blocks)],
    )


def make_table_from_cells(
    bbox: BBox,
    cells: list[LayoutBlock],
    *,
    prefix: str = "t",
) -> LayoutBlock:
    """已带独立 bbox 的格子（含合并单元格）。"""
    return LayoutBlock(id=prefix, type="table", bbox=bbox.clamped(), cells=list(cells))


def tag_running_regions(blocks: list[LayoutBlock]) -> None:
    """页眉页脚：贴在纸顶/纸底的矮块。"""
    for block in blocks:
        if block.type in ("table", "cell"):
            continue
        box = block.bbox.clamped()
        if box.y + box.h <= 9.5 and box.h <= 12:
            block.type = "header"
            block.style.indent = 0
        elif box.y >= 90.5 and box.h <= 12:
            block.type = "footer"
            block.style.indent = 0


def assign_columns(blocks: list[LayoutBlock]) -> bool:
    """两栏：中间有空隙、左右都有足够块时打 column=0/1。"""
    body = [b for b in blocks if b.type not in ("header", "footer", "table")]
    if len(body) < 4:
        return False
    left = [b for b in body if (b.bbox.x + b.bbox.w / 2.0) < 46]
    right = [b for b in body if (b.bbox.x + b.bbox.w / 2.0) > 54]
    mid = [b for b in body if 46 <= (b.bbox.x + b.bbox.w / 2.0) <= 54]
    if len(left) < 2 or len(right) < 2 or len(mid) > 1:
        return False
    for block in left:
        block.column = 0
    for block in right:
        block.column = 1
    return True


def make_table_grid(
    bbox: BBox,
    rows: int,
    cols: int,
    *,
    prefix: str = "t",
    texts: list[str] | None = None,
) -> LayoutBlock:
    rows = max(1, min(12, int(rows)))
    cols = max(1, min(8, int(cols)))
    cells = _grid_cells(bbox, rows, cols, prefix=prefix, texts=texts or [])
    return LayoutBlock(id=prefix, type="table", bbox=bbox, cells=cells)


def _grid_cells(
    bbox: BBox,
    rows: int,
    cols: int,
    *,
    prefix: str,
    texts: list[str],
) -> list[LayoutBlock]:
    box = bbox.clamped()
    cell_w = box.w / cols
    cell_h = box.h / rows
    cells: list[LayoutBlock] = []
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            text = texts[index] if index < len(texts) else ""
            cells.append(
                LayoutBlock(
                    id=f"{prefix}-r{row}c{col}",
                    type="cell",
                    bbox=BBox(
                        x=box.x + col * cell_w,
                        y=box.y + row * cell_h,
                        w=cell_w,
                        h=cell_h,
                    ),
                    text=str(text or ""),
                    style=BlockStyle(indent=0),
                )
            )
    return cells


def single_page_doc(
    text: str,
    *,
    bbox: BBox | None = None,
    width_mm: float = 210.0,
    height_mm: float = 297.0,
    align: str = "left",
    indent: int = 2,
    background: dict[str, Any] | None = None,
) -> LayoutDoc:
    """旧的单区域请求：整页（或一个框）一块。"""
    return LayoutDoc(
        source={"kind": "plain"},
        pages=[
            LayoutPage(
                width_mm=width_mm,
                height_mm=height_mm,
                background=background,
                blocks=[
                    LayoutBlock(
                        id="r0",
                        type="box",
                        bbox=bbox or BBox(),
                        text=text,
                        style=BlockStyle(align=align if align in ALIGN_VALUES else "left", indent=indent),
                    )
                ],
            )
        ],
    )


@dataclass(frozen=True)
class BlockSpec:
    """渲染器内部用的扁平块（表格已拆格）。"""

    text: str
    x: float
    y: float
    w: float
    h: float
    align: Literal["left", "center", "right"] = "left"
    indent: int = 0
    glyph_scale: float = 1.0
    erase: bool = False
    style_id: str = ""


def _cap_glyph_scale(scale: float, box: BBox, text: str) -> float:
    """矮框按行高封顶，避免手写字撑出文字框。"""
    explicit_lines = max(1, (text or "").count("\n") + 1)
    chars = len((text or "").replace("\n", ""))
    chars_per_line = max(4.0, box.w / 3.5)
    wrapped = max(1, int((chars + chars_per_line - 1) // chars_per_line)) if chars else 1
    line_count = max(explicit_lines, wrapped)
    line_h_pct = 4.0
    max_scale = (box.h / line_count) / line_h_pct
    return max(0.4, min(scale, max_scale))


def flatten_pages(doc: LayoutDoc, *, default_font_size: int = 24) -> list[list[BlockSpec]]:
    pages: list[list[BlockSpec]] = []
    for page in doc.pages:
        specs: list[BlockSpec] = []
        for block in iter_leaf_blocks(page.blocks):
            box = block.bbox.clamped()
            if box.w < MIN_BOX_W or box.h < MIN_BOX_H:
                continue
            font_size = block.style.fontSize or default_font_size
            scale = max(0.55, min(2.2, float(font_size) / max(1, default_font_size)))
            if block.type == "heading":
                scale = max(scale, 1.15)
            if block.type in ("header", "footer"):
                scale = min(scale, 0.85)
            scale = _cap_glyph_scale(scale, box, block.text or "")
            align = block.style.align if block.style.align in ALIGN_VALUES else "left"
            specs.append(
                BlockSpec(
                    text=block.text or "",
                    x=box.x,
                    y=box.y,
                    w=box.w,
                    h=box.h,
                    align=align,  # type: ignore[arg-type]
                    indent=0 if block.type in ("header", "footer") else block.style.indent,
                    glyph_scale=scale,
                    erase=block.erase,
                    style_id=block.style.styleId or "",
                )
            )
        pages.append(specs)
    return pages


def specs_to_config_pages(pages: list[list[BlockSpec]]) -> tuple:
    """RenderConfig.layout_pages 只收可冻结的 tuple[dict]。"""
    return tuple(
        tuple(
            {
                "text": spec.text,
                "x": spec.x,
                "y": spec.y,
                "w": spec.w,
                "h": spec.h,
                "align": spec.align,
                "indent": spec.indent,
                "glyph_scale": spec.glyph_scale,
                "erase": spec.erase,
                "style_id": spec.style_id,
            }
            for spec in page
        )
        for page in pages
    )
