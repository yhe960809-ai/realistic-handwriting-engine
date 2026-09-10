"""整页渲染管线：排版 → 逐字合成 → 墨迹沉积 → 纸面合成 → 成像仿真。

所有几何量以毫米为单位，像素只在最后一步出现（开发原则 #3）。
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import stamp, supplement
from ..progress import report_page
from .coverage import _is_cjk_ideograph, missing_message, normalize_text, unsupported_clusters
from .ink import apply_absorbency_bleed, composite, deposit, srgb_to_linear
from .legacy import USE_LEGACY_INK
from .paper import CaptureSpec, PaperSpec, apply_capture, make_paper
from .skeleton import GlyphSkeleton, load_bank
from .style import Pen, WriterStyle, get_style, has_style
from .trajectory import synthesize

# 逐页成像的并发度。这一段是 numpy / PIL 的整页运算，会释放 GIL，实测 4 路
# 并发把 3000 字从 16.0s 压到 11.0s。上限由内存定：每页峰值几百 MB，要守住
# 方案 §1.3 的 600MB/请求 预算。
#
# 逐字合成没有对应的线程池：那一段以 Python 层的轨迹合成为主，GIL 让它无法
# 真正并行，实测 2 线程比 1 线程慢 19%，4 线程慢 34%。已删除。
_PAGE_WORKERS = 4

# 避头标点：不能出现在行首
_NO_LINE_START = "，。、；：？！”’）》】…·,.;:!?)》]}>」』〕〗"
# 避尾标点：不能出现在行末
_NO_LINE_END = "“‘（《【(（[{<「『〔〖"
# 标点相对汉字的字身占位
_PUNCT_ADVANCE = 0.58
# ASCII 空格按汉字宽会把英文词距撑得过大
_SPACE_ADVANCE = {" ": 0.32, "\t": 0.32, "\u00a0": 0.32, "\u3000": 1.0}
_PAGE_BREAK = "\f"


@dataclass
class PageSpec:
    """页面几何。默认 A4。"""

    width_mm: float = 210.0
    height_mm: float = 297.0
    dpi: float = 300.0
    margin_left_mm: float = 22.0
    margin_right_mm: float = 18.0
    margin_top_mm: float = 24.0
    margin_bottom_mm: float = 22.0
    # 汉字字身高度（毫米）。真人日常书写约 6~10mm
    glyph_size_mm: float = 7.4
    # 行距（毫米）。留空则按字身的 1.62 倍
    line_pitch_mm: float | None = None
    # 字距附加量（相对字身）
    tracking: float = 0.06
    # 段首缩进（字数）
    indent: int = 2
    # 字身底边相对字身高度的位置。汉字写在横线上时底边压线，这个值决定压多少。
    baseline_ratio: float = 0.90
    # 有横线/网格时，行位置必须由纸张的真实线位决定，而不是自己算一套
    bind_to_ruling: bool = True
    # 绑定横线时，字身高度占行距的比例。由 tools/calibrate_glyph_ratio.py 回环
    # 标定：真人语料（47 张横格/方格手写照）墨迹高度占行距中位数 0.680，按本值
    # 渲染再用同一把尺子量回去，0.87 对应 0.667、0.98 对应 0.763。原值 0.74 量出
    # 0.588 —— 字比真人小了一圈，一行塞不满，整页显得空。
    glyph_to_rule_ratio: float = 0.87
    # 照片纸面检测出的真实基线（像素）。有值时压过一切推算
    forced_baseline_ys: list[float] | None = None

    @property
    def px_per_mm(self) -> float:
        return self.dpi / 25.4

    @property
    def width_px(self) -> int:
        return int(round(self.width_mm * self.px_per_mm))

    @property
    def height_px(self) -> int:
        return int(round(self.height_mm * self.px_per_mm))

    @property
    def resolved_line_pitch_mm(self) -> float:
        return self.line_pitch_mm if self.line_pitch_mm else self.glyph_size_mm * 1.62


@dataclass
class RenderRequest:
    text: str
    style_id: str = "xingkai-daily"
    page: PageSpec = field(default_factory=PageSpec)
    paper: PaperSpec = field(default_factory=PaperSpec)
    capture: CaptureSpec = field(default_factory=CaptureSpec)
    seed: int = 20260823
    supersample: int = 3
    # 普通字的书写实例数。高频汉字会按出现次数升到 8–12，见 _variant_budget。
    variants_per_char: int = 4
    # 只出前 N 页。预览场景下用户看的是第一页长什么样，把后面几十页也画完
    # 是纯粹的浪费；metadata 里仍然报告完整页数，UI 可以照实说"共 N 页"。
    max_pages: int | None = None
    # 多块版式：每一页一组 BlockSpec。有值时不再把 text 整页回流。
    layout_pages: list[list[BlockSpec]] | None = None
    text_align: str = "left"
    # 覆盖风格自带笔。None = 不换笔。整页（含多块不同 style_id）共用这一支。
    pen: Pen | None = None
    scribble_count: int = 0
    scribble_style: str = "strike"
    # expand 写入：flow 用全文下标；多块用与 layout_pages 平行的 frozenset 列表。
    scribble_at: frozenset[int] = field(default_factory=frozenset)
    scribble_layout: list[list[frozenset[int]]] | None = None


@dataclass
class RenderResult:
    images: list[np.ndarray]
    metadata: dict


def _drift_series(count: int, sigma: float, rho: float, fatigue: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) 漂移叠加线性疲劳趋势。真人写长文的变化是低频相关的。"""
    if count <= 0:
        return np.zeros(0)
    innovation = rng.normal(0.0, sigma * math.sqrt(max(1e-9, 1.0 - rho**2)), size=count)
    out = np.empty(count)
    value = rng.normal(0.0, sigma)
    for index in range(count):
        value = rho * value + innovation[index]
        out[index] = value
    if fatigue:
        out += np.linspace(0.0, fatigue, count)
    return out


def _advance_em(char: str) -> float:
    # 步进以字形自己声明的为准：标点的墨迹位置与留白都在 supplement 里定死，
    # 这里再写一个常量就是第二份真相，两边一改就错位。
    if char in _SPACE_ADVANCE:
        return _SPACE_ADVANCE[char]
    if char == _PAGE_BREAK:
        return 0.0
    if supplement.has(char) and not ("\u4e00" <= char <= "\u9fff"):
        return supplement.advance_of(char)
    if char in _NO_LINE_START or char in _NO_LINE_END:
        return _PUNCT_ADVANCE
    return 1.0


def _plain_char(item) -> str:
    if isinstance(item, tuple):
        return item[0]
    return item


def _as_flow_line(line) -> tuple[list[str], str]:
    if isinstance(line, tuple):
        raw, align = list(line[0]), str(line[1] or "left")
    else:
        raw, align = list(line), "left"
    return [_plain_char(c) for c in raw], align


def _flagged_items(line) -> list[tuple[str, bool]]:
    if isinstance(line, tuple):
        raw = list(line[0])
    else:
        raw = list(line)
    out: list[tuple[str, bool]] = []
    for item in raw:
        if isinstance(item, tuple):
            out.append((item[0], bool(item[1])))
        else:
            out.append((item, False))
    return out


def _is_page_break(line) -> bool:
    chars, _align = _as_flow_line(line)
    return chars == [_PAGE_BREAK]


def _page_count(lines: list, lines_per_page: int) -> int:
    pages = 1
    used = 0
    for line in lines:
        if _is_page_break(line):
            if used:
                pages += 1
                used = 0
            continue
        if used >= lines_per_page:
            pages += 1
            used = 0
        used += 1
    return pages


def _take_pages(lines: list, lines_per_page: int, max_pages: int) -> list:
    out: list = []
    pages = 1
    used = 0
    for line in lines:
        if _is_page_break(line):
            if not used:
                continue
            if pages >= max_pages:
                break
            out.append(line)
            pages += 1
            used = 0
            continue
        if used >= lines_per_page:
            if pages >= max_pages:
                break
            pages += 1
            used = 0
        out.append(line)
        used += 1
    return out


def _wrap(text: str, columns_em: float, indent: int) -> list[tuple[list[str], str]]:
    """按字身宽度贪心折行，遵守中文避头避尾规则。

    每行是 (字符列表, 对齐)。``>>>`` 行右对齐；独占一行的 ``---`` 是手动分页。
    """
    lines: list[tuple[list[str], str]] = []
    for paragraph in text.split("\n"):
        stripped = paragraph.strip()
        if not stripped:
            lines.append(([], "left"))
            continue
        if len(stripped) >= 3 and set(stripped) <= {"-"}:
            lines.append(([_PAGE_BREAK], "left"))
            continue
        align = "left"
        if stripped.startswith(">>>"):
            stripped = stripped[3:].strip()
            align = "right"
            if not stripped:
                lines.append(([], align))
                continue
        prefix = "" if align == "right" else ("\u3000" * indent)
        chars = list(prefix + stripped)
        current: list[str] = []
        used = 0.0
        for char in chars:
            width = _advance_em(char)
            if used + width > columns_em + 1e-6 and current:
                # 避头：行首标点回拉到上一行
                if char in _NO_LINE_START:
                    current.append(char)
                    lines.append((current, align))
                    current, used = [], 0.0
                    continue
                # 避尾：行末的开引号/开括号推到下一行
                if current[-1] in _NO_LINE_END:
                    trailing = current.pop()
                    lines.append((current, align))
                    current = [trailing, char]
                    used = _advance_em(trailing) + width
                    continue
                lines.append((current, align))
                current, used = [char], width
                continue
            current.append(char)
            used += width
        if current:
            lines.append((current, align))
    return lines


def _wrap_flagged(
    text: str, scribble_at: frozenset[int], columns_em: float, indent: int
) -> list[tuple[list[tuple[str, bool]], str]]:
    """与 ``_wrap`` 同一套折行，并带着 expand 留下的涂改下标。"""
    marked = scribble_at or frozenset()
    lines: list[tuple[list[tuple[str, bool]], str]] = []
    offset = 0
    parts = text.split("\n")
    for para_i, paragraph in enumerate(parts):
        if para_i:
            offset += 1
        para_start = offset
        offset += len(paragraph)
        left = len(paragraph) - len(paragraph.lstrip())
        right = len(paragraph.rstrip())
        stripped = paragraph[left:right]
        if not stripped:
            lines.append(([], "left"))
            continue
        if len(stripped) >= 3 and set(stripped) <= {"-"}:
            lines.append(([(_PAGE_BREAK, False)], "left"))
            continue
        align = "left"
        body = stripped
        body_base = para_start + left
        if stripped.startswith(">>>"):
            after = stripped[3:]
            rest_left = len(after) - len(after.lstrip())
            rest = after.strip()
            align = "right"
            if not rest:
                lines.append(([], align))
                continue
            body = rest
            body_base = para_start + left + 3 + rest_left
        flagged = [(ch, (body_base + j) in marked) for j, ch in enumerate(body)]
        prefix: list[tuple[str, bool]] = (
            [] if align == "right" else [("\u3000", False)] * indent
        )
        chars = prefix + flagged
        current: list[tuple[str, bool]] = []
        used = 0.0
        for item in chars:
            char = item[0]
            width = _advance_em(char)
            if used + width > columns_em + 1e-6 and current:
                if char in _NO_LINE_START:
                    current.append(item)
                    lines.append((current, align))
                    current, used = [], 0.0
                    continue
                if current[-1][0] in _NO_LINE_END:
                    trailing = current.pop()
                    lines.append((current, align))
                    current = [trailing, item]
                    used = _advance_em(trailing[0]) + width
                    continue
                lines.append((current, align))
                current, used = [item], width
                continue
            current.append(item)
            used += width
        if current:
            lines.append((current, align))
    return lines


def _lines_for_text(
    text: str, columns_em: float, indent: int, scribble_at: frozenset[int] | None
) -> list:
    marks = scribble_at or frozenset()
    if not marks:
        return _wrap(text, columns_em, indent)
    return _wrap_flagged(text, marks, columns_em, indent)


# 一篇里只给真正高频的汉字加变体，避免「整段复制 60 遍」的压测把每个字都抬上去。
_HIGH_FREQ_MIN_COUNT = 8
_HIGH_FREQ_TOP_K = 8
_HIGH_FREQ_MIN_VARIANTS = 8
_HIGH_FREQ_MAX_VARIANTS = 12


def _is_hanzi(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _high_freq_variants(count: int, ordinary: int) -> int:
    """Map occurrence count to 8–12 (never below the ordinary budget)."""
    lo = max(ordinary, _HIGH_FREQ_MIN_VARIANTS)
    cap = max(ordinary, _HIGH_FREQ_MAX_VARIANTS)
    if count >= 32:
        return cap
    if count >= 16:
        return min(cap, max(lo, 10))
    return min(cap, lo)


def variant_budget(counts: dict[str, int], ordinary: int = 4) -> dict[str, int]:
    """Per-character variant caps: 4 for ordinary glyphs, 8–12 for top frequent hanzi."""
    ordinary = max(1, ordinary)
    ranked = sorted(
        (
            (char, n)
            for char, n in counts.items()
            if n >= _HIGH_FREQ_MIN_COUNT and _is_hanzi(char)
        ),
        key=lambda item: (-item[1], item[0]),
    )[:_HIGH_FREQ_TOP_K]
    boosted = {char for char, _ in ranked}
    return {
        char: _high_freq_variants(n, ordinary) if char in boosted else ordinary
        for char, n in counts.items()
    }


def _reuse_report(
    counts: dict[str, int], budgets: dict[str, int], jobs: dict[tuple, tuple]
) -> dict[str, dict]:
    rendered: dict[str, int] = Counter(key[0] for key in jobs)
    report: dict[str, dict] = {}
    for char, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count <= 0:
            continue
        variants = int(rendered.get(char, 0))
        report[char] = {
            "count": count,
            "budget": int(budgets.get(char, 4)),
            "variants": variants,
            "reused": max(0, count - variants),
            "reuse_rate": round(max(0, count - variants) / count, 4),
        }
    return report


def _resolve_skeleton(char: str) -> tuple[GlyphSkeleton | None, str]:
    bank = load_bank()
    skeleton = bank.get(char)
    if skeleton is not None:
        return skeleton, "bank"
    skeleton = supplement.get(char)
    if skeleton is not None:
        return skeleton, "supplement"
    return None, "missing"


def _resolved_glyph_mm(request: RenderRequest) -> float:
    page = request.page
    glyph_mm = page.glyph_size_mm
    if page.bind_to_ruling and request.paper.ruling in ("lined", "grid"):
        glyph_mm = request.paper.line_pitch_mm * page.glyph_to_rule_ratio
    return glyph_mm


def _block_would_clip(text: str, spec, request: RenderRequest) -> bool:
    """与 ``_place_layout_pages`` 同一套折行/缩小；缩小后仍超行则视为会裁字。"""
    page = request.page
    px_per_mm = page.px_per_mm
    width_px, height_px = page.width_px, page.height_px
    glyph_mm = _resolved_glyph_mm(request)
    pitch_px = page.resolved_line_pitch_mm * px_per_mm
    box_width = width_px * spec.w / 100.0
    box_top = height_px * spec.y / 100.0
    box_bottom = box_top + height_px * spec.h / 100.0
    local_glyph_mm = max(3.2, glyph_mm * spec.glyph_scale)
    local_em = local_glyph_mm * px_per_mm
    local_pitch = max(local_em * 1.15, pitch_px * spec.glyph_scale)
    columns_em = max(1.0, (box_width / px_per_mm) / (local_glyph_mm * (1.0 + page.tracking)))
    lines = _wrap(text, columns_em, spec.indent)
    baselines = _box_baselines(box_top, box_bottom, local_pitch, page.forced_baseline_ys)
    if len(lines) <= len(baselines):
        return False
    shrink = max(0.58, len(baselines) / max(1, len(lines)))
    local_glyph_mm = max(3.2, local_glyph_mm * shrink)
    local_em = local_glyph_mm * px_per_mm
    local_pitch = max(local_em * 1.15, local_pitch * shrink)
    columns_em = max(1.0, (box_width / px_per_mm) / (local_glyph_mm * (1.0 + page.tracking)))
    lines = _wrap(text, columns_em, spec.indent)
    baselines = _box_baselines(box_top, box_bottom, local_pitch, page.forced_baseline_ys)
    return len(lines) > len(baselines)


def _apply_mistakes(request: RenderRequest) -> None:
    from .mistakes import STYLES, expand, expand_blocks

    if request.scribble_style not in STYLES:
        request.scribble_style = "strike"
    if request.layout_pages:
        specs = [spec for page in request.layout_pages for spec in page]
        texts = [spec.text for spec in specs]

        def fits(idx: int, new_text: str) -> bool:
            return not _block_would_clip(new_text, specs[idx], request)

        new_texts, ats = expand_blocks(
            texts, request.scribble_count, request.seed, fits=fits
        )
        text_iter = iter(new_texts)
        at_iter = iter(ats)
        new_pages = []
        layout_marks = []
        for page in request.layout_pages:
            page_specs = []
            page_marks = []
            for spec in page:
                page_specs.append(dataclasses.replace(spec, text=next(text_iter)))
                page_marks.append(next(at_iter))
            new_pages.append(page_specs)
            layout_marks.append(page_marks)
        request.layout_pages = new_pages
        request.scribble_layout = layout_marks
        return
    request.text, request.scribble_at = expand(
        request.text, request.scribble_count, request.seed
    )


def _prepare_request(request: RenderRequest) -> None:
    """归一手机 Emoji 序列，并在开渲前报缺字，避免白渲一整页。"""
    request.text = normalize_text(request.text)
    if request.layout_pages:
        request.layout_pages = [
            [dataclasses.replace(block, text=normalize_text(block.text)) for block in page]
            for page in request.layout_pages
        ]
    if int(getattr(request, "scribble_count", 0) or 0) > 0:
        _apply_mistakes(request)
    scan_source = request.text
    if request.layout_pages:
        scan_source += "".join(block.text for page in request.layout_pages for block in page)
    missing = unsupported_clusters(scan_source)
    if missing:
        raise ValueError(missing_message(missing))


def render(request: RenderRequest) -> RenderResult:
    started = time.perf_counter()
    _prepare_request(request)
    page = request.page
    style = get_style(request.style_id)
    if request.pen is not None:
        style = style.with_pen(request.pen)
    px_per_mm = page.px_per_mm
    glyph_mm = page.glyph_size_mm
    if page.bind_to_ruling and request.paper.ruling in ("lined", "grid"):
        # 写在横线纸上时，字身高度由行宽决定 —— 真人不会写出跨线的字
        glyph_mm = request.paper.line_pitch_mm * page.glyph_to_rule_ratio
    em_px = glyph_mm * px_per_mm
    pitch_px = page.resolved_line_pitch_mm * px_per_mm

    load_bank()
    occurrence_counter: dict[str, int] = {}
    fallbacks: list[str] = []
    missing: list[str] = []
    placements: list[tuple[int, tuple, float, float, int]] = []
    jobs: dict[tuple, tuple] = {}
    width_px, height_px = page.width_px, page.height_px

    if request.layout_pages:
        total_pages = max(1, len(request.layout_pages))
        page_budget = min(total_pages, request.max_pages or total_pages)
        visible_text = "\n".join(
            spec.text
            for page_blocks in request.layout_pages[:page_budget]
            for spec in page_blocks
        )
        visible_counts = Counter(char for char in visible_text if not char.isspace())
        total_chars = max(1, sum(visible_counts.values()) + visible_text.count("\n") + 1)
        variant_budgets = variant_budget(visible_counts, request.variants_per_char)
        char_index = _place_layout_pages(
            request,
            glyph_mm,
            em_px,
            pitch_px,
            px_per_mm,
            page_budget,
            placements,
            jobs,
            occurrence_counter,
            fallbacks,
            missing,
            variant_budgets,
        )
        page_i = max(0, page_budget - 1)
        coverage_text = visible_text or request.text
    else:
        usable_mm = page.width_mm - page.margin_left_mm - page.margin_right_mm
        columns_em = usable_mm / (glyph_mm * (1.0 + page.tracking))
        lines = _lines_for_text(request.text, columns_em, page.indent, request.scribble_at)

        # 行位置用独立的随机流：它决定分页，不能被正文长度反过来影响。
        layout_rng = np.random.default_rng((request.seed ^ 0x9E3779B9) & 0xFFFFFFFF)
        baseline_ys = _baseline_positions(
            page,
            request.paper,
            px_per_mm,
            pitch_px,
            pitch_jitter=style.drift.line_pitch_sigma,
            rng=layout_rng,
        )
        lines_per_page = max(1, len(baseline_ys))
        total_pages = _page_count(lines, lines_per_page)

        page_budget = request.max_pages or total_pages
        visible_lines = _take_pages(lines, lines_per_page, page_budget)
        visible_counts = Counter(
            char
            for line in visible_lines
            for char in _as_flow_line(line)[0]
            if not char.isspace() and char != _PAGE_BREAK
        )
        total_chars = max(
            1,
            sum(len(_as_flow_line(line)[0]) for line in lines if not _is_page_break(line)),
        )
        variant_budgets = variant_budget(visible_counts, request.variants_per_char)
        char_index = _place_flow_lines(
            request,
            visible_lines,
            baseline_ys,
            lines_per_page,
            em_px,
            placements,
            jobs,
            occurrence_counter,
            fallbacks,
            missing,
            variant_budgets,
            total_chars,
        )
        page_i = 0
        if placements:
            page_i = max(item[0] for item in placements)
        coverage_text = request.text

    seed_rng = np.random.default_rng(request.seed & 0xFFFFFFFF)
    drift = style.drift
    size_drift = _drift_series(total_chars, drift.size_sigma, drift.rho, drift.fatigue_size, seed_rng)
    slant_drift = _drift_series(total_chars, drift.slant_sigma, drift.rho, drift.fatigue_slant, seed_rng)
    baseline_drift = _drift_series(total_chars, drift.baseline_sigma, drift.rho, 0.0, seed_rng)
    pressure_drift = _drift_series(total_chars, drift.pressure_sigma, drift.rho, 0.0, seed_rng)

    def _render_glyph(item):
        cache_key, char, variant, index, skeleton, local_em, style_id, source = item
        used = get_style(style_id) if style_id and has_style(style_id) else style
        if request.pen is not None:
            used = used.with_pen(request.pen)
        local_style = _drifted_style(used, slant_drift[index], pressure_drift[index])
        if source == "font-stamp":
            return cache_key, stamp.deposit_mass(
                char,
                em_px=local_em,
                slant=local_style.structure.slant,
                aspect=local_style.structure.aspect,
                gravity_dx=local_style.structure.gravity_dx,
                gravity_dy=local_style.structure.gravity_dy,
                supersample=request.supersample,
                variant=variant,
                seed=request.seed,
            )
        paths = synthesize(
            skeleton, local_style, seed=request.seed, occurrence=variant, em_px=local_em
        )
        return cache_key, deposit(
            paths,
            em_px=local_em,
            pen=local_style.pen,
            px_per_mm=px_per_mm,
            supersample=request.supersample,
            seed=_glyph_seed(request.seed, char, variant),
        )

    glyph_cache: dict[tuple, tuple[np.ndarray, tuple[int, int]]] = {}
    for item in ((k, *v) for k, v in jobs.items()):
        key, rendered = _render_glyph(item)
        glyph_cache[key] = rendered

    page_count = page_i + 1
    pages_mass = [np.zeros((height_px, width_px), dtype=np.float32) for _ in range(page_count)]
    for page_i, cache_key, pen_x, baseline_y, index, stamp_em in placements:
        glyph_mass, offset = glyph_cache[cache_key]
        size_scale = 1.0 + float(size_drift[index])
        if abs(size_scale - 1.0) > 0.012:
            glyph_mass, offset = _scale_stamp(glyph_mass, offset, size_scale)
        origin_x = int(round(pen_x + offset[0]))
        origin_y = int(
            round(
                baseline_y
                - page.baseline_ratio * stamp_em
                + offset[1]
                + baseline_drift[index] * stamp_em
            )
        )
        _accumulate(pages_mass[page_i], glyph_mass, origin_x, origin_y)

    paper_linear, absorbency = make_paper(
        width_px, height_px, request.paper, px_per_mm, request.seed
    )
    # 页级分辨率上的纤维洇散尺度。deposit 已在超采样画布上做过匀质洇散，
    # 这里只让吸水率偏离 0.5 的地方再拉开，中性纸面保持恒等。
    absorbency_sigma = style.pen.bleed_sigma_px * max(px_per_mm / 10.0, 0.4)

    def _finish_page(item: tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
        page_index, page_mass = item
        if not USE_LEGACY_INK:
            page_mass = apply_absorbency_bleed(
                page_mass, absorbency, sigma=absorbency_sigma
            )
        blended = composite(paper_linear, page_mass, style.pen)
        return page_index, apply_capture(blended, request.capture, request.seed + page_index)

    # 成像是逐页且最贵的一段，进度与超时检查都挂在这里。页与页之间没有依赖，
    # 但每页要占几百 MB 中间数组，所以并发数由内存预算而不是核数决定。
    images: list[np.ndarray] = [None] * len(pages_mass)  # type: ignore[list-item]
    page_workers = min(_PAGE_WORKERS, len(pages_mass))
    if page_workers <= 1:
        for item in enumerate(pages_mass):
            index, image = _finish_page(item)
            images[index] = image
            report_page(index + 1, len(pages_mass))
    else:
        finished = 0
        with ThreadPoolExecutor(max_workers=page_workers) as pool:
            for index, image in pool.map(_finish_page, enumerate(pages_mass)):
                images[index] = image
                finished += 1
                report_page(finished, len(pages_mass))

    bank = load_bank()
    hit, total, miss = bank.coverage(coverage_text)
    metadata = {
        "engine": "v3-zhenji",
        "engine_version": "3.0.0",
        "skeleton_bank": bank.bank_id,
        "style_id": style.style_id,
        "pen_id": style.pen.pen_id,
        "seed": request.seed,
        "pages": len(images),
        # 实际排完全文需要多少页。截页预览时与 pages 不同。
        "total_pages": total_pages,
        "truncated": len(images) < total_pages,
        "glyphs_rendered": char_index,
        "distinct_glyph_renders": len(glyph_cache),
        "variant_reuse": _reuse_report(dict(visible_counts), variant_budgets, jobs),
        "bank_coverage": {"hit": hit, "total": total, "missing": miss},
        "fallbacks": sorted(set(fallbacks)),
        "font_stamp_glyphs": sorted(
            {entry.split(":", 1)[0] for entry in fallbacks if ":font-stamp" in entry}
        ),
        "missing_glyphs": sorted(set(missing)),
        "dpi": page.dpi,
        "px_per_mm": round(px_per_mm, 4),
        "elapsed_s": round(time.perf_counter() - started, 3),
        # AIGC 标识不在这里生成：它必须写进最终交付的文件字节，而 v3 只产出
        # 中间位图。实现见 renderer.aigc，由 core.render_with_metadata 落盘。
    }
    return RenderResult(images=images, metadata=metadata)


def _place_glyph(
    char: str,
    *,
    char_index: int,
    total_chars: int,
    page_i: int,
    pen_x: float,
    baseline_y: float,
    em_px: float,
    tracking: float,
    placements: list,
    jobs: dict,
    occurrence_counter: dict[str, int],
    fallbacks: list[str],
    missing: list[str],
    variant_budgets: dict[str, int],
    variants_per_char: int,
    style_id: str = "",
) -> float:
    advance = _advance_em(char)
    step = advance * em_px * (1.0 + tracking)
    if char.isspace():
        return step
    skeleton, source = _resolve_skeleton(char)
    if skeleton is None:
        font_id = (
            stamp.resolve(char)
            if stamp.enabled() and _is_cjk_ideograph(char)
            else None
        )
        if font_id is None:
            missing.append(char)
            return step
        source = "font-stamp"
        fallbacks.append(f"{char}:font-stamp:{font_id}")
    elif source != "bank":
        fallbacks.append(f"{char}:{source}")
    index = min(char_index, total_chars - 1)
    occurrence = occurrence_counter.get(char, 0)
    occurrence_counter[char] = occurrence + 1
    if source == "font-stamp":
        variant = occurrence % stamp.VARIANT_COUNT
    else:
        budget = variant_budgets.get(char, variants_per_char)
        variant = occurrence % max(1, budget)
    cache_key = (char, variant, round(em_px, 1), style_id or "")
    if cache_key not in jobs:
        jobs[cache_key] = (char, variant, index, skeleton, em_px, style_id or "", source)
    placements.append((page_i, cache_key, pen_x, baseline_y, index, em_px))
    return step


def _add_scratch_placement(
    *,
    page_i: int,
    pen_x: float,
    baseline_y: float,
    em_px: float,
    index: int,
    placements: list,
    jobs: dict,
    scribble_style: str,
) -> None:
    from .mistakes import scratch_skeleton

    kind = scribble_style if scribble_style in ("strike", "scribble", "cross") else "strike"
    key_char = f"\x00scratch:{kind}"
    cache_key = (key_char, 0, round(em_px, 1), "")
    if cache_key not in jobs:
        jobs[cache_key] = (key_char, 0, index, scratch_skeleton(kind), em_px, "", "supplement")
    placements.append((page_i, cache_key, pen_x, baseline_y, index, em_px))


def _layout_scribble_at(request: RenderRequest, page_i: int, block_i: int) -> frozenset[int]:
    rows = request.scribble_layout
    if not rows or page_i >= len(rows):
        return frozenset()
    row = rows[page_i]
    if block_i >= len(row):
        return frozenset()
    return row[block_i]


def _line_start_x(
    line: list[str],
    box_left: float,
    box_width: float,
    em_px: float,
    tracking: float,
    align: str,
) -> float:
    if align not in ("center", "right"):
        return box_left
    width = sum(_advance_em(char) * em_px * (1.0 + tracking) for char in line)
    leftover = max(0.0, box_width - width)
    if align == "center":
        return box_left + leftover / 2.0
    return box_left + leftover


def _place_flow_lines(
    request: RenderRequest,
    visible_lines: list[list[str]],
    baseline_ys: list[float],
    lines_per_page: int,
    em_px: float,
    placements: list,
    jobs: dict,
    occurrence_counter: dict[str, int],
    fallbacks: list[str],
    missing: list[str],
    variant_budgets: dict[str, int],
    total_chars: int,
) -> int:
    page = request.page
    px_per_mm = page.px_per_mm
    box_left = page.margin_left_mm * px_per_mm
    box_width = (page.width_mm - page.margin_left_mm - page.margin_right_mm) * px_per_mm
    char_index = 0
    line_on_page = 0
    page_i = 0
    for raw_line in visible_lines:
        line, line_align = _as_flow_line(raw_line)
        if line == [_PAGE_BREAK]:
            if line_on_page:
                page_i += 1
                line_on_page = 0
            continue
        if line_on_page >= lines_per_page:
            page_i += 1
            line_on_page = 0
        baseline_y = baseline_ys[line_on_page]
        align = line_align if line_align in ("left", "center", "right") else request.text_align
        pen_x = _line_start_x(line, box_left, box_width, em_px, page.tracking, align)
        for char, scratch in _flagged_items(raw_line):
            origin = pen_x
            pen_x += _place_glyph(
                char,
                char_index=char_index,
                total_chars=total_chars,
                page_i=page_i,
                pen_x=origin,
                baseline_y=baseline_y,
                em_px=em_px,
                tracking=page.tracking,
                placements=placements,
                jobs=jobs,
                occurrence_counter=occurrence_counter,
                fallbacks=fallbacks,
                missing=missing,
                variant_budgets=variant_budgets,
                variants_per_char=request.variants_per_char,
            )
            if scratch:
                _add_scratch_placement(
                    page_i=page_i,
                    pen_x=origin,
                    baseline_y=baseline_y,
                    em_px=em_px,
                    index=min(char_index, total_chars - 1),
                    placements=placements,
                    jobs=jobs,
                    scribble_style=request.scribble_style,
                )
            char_index += 1
        line_on_page += 1
    return char_index


def _box_baselines(
    top_px: float,
    bottom_px: float,
    pitch_px: float,
    forced: list[float] | None,
) -> list[float]:
    if forced:
        inside = [float(y) for y in forced if top_px + pitch_px * 0.25 <= float(y) <= bottom_px]
        if inside:
            return inside
    positions: list[float] = []
    y = top_px + pitch_px
    while y <= bottom_px:
        positions.append(float(y))
        y += pitch_px
    return positions or [min(bottom_px, top_px + pitch_px)]


def _place_layout_pages(
    request: RenderRequest,
    glyph_mm: float,
    em_px: float,
    pitch_px: float,
    px_per_mm: float,
    page_budget: int,
    placements: list,
    jobs: dict,
    occurrence_counter: dict[str, int],
    fallbacks: list[str],
    missing: list[str],
    variant_budgets: dict[str, int],
) -> int:
    page = request.page
    width_px, height_px = page.width_px, page.height_px
    char_index = 0
    total_chars = max(
        1,
        sum(len(spec.text) for page_blocks in request.layout_pages[:page_budget] for spec in page_blocks),
    )
    for page_i, page_blocks in enumerate(request.layout_pages[:page_budget]):
        for block_i, spec in enumerate(page_blocks):
            if not (spec.text or "").strip():
                continue
            box_left = width_px * spec.x / 100.0
            box_top = height_px * spec.y / 100.0
            box_width = width_px * spec.w / 100.0
            box_height = height_px * spec.h / 100.0
            box_right = box_left + box_width
            box_bottom = box_top + box_height
            local_glyph_mm = max(3.2, glyph_mm * spec.glyph_scale)
            local_em = local_glyph_mm * px_per_mm
            local_pitch = max(local_em * 1.15, pitch_px * spec.glyph_scale)
            columns_em = max(1.0, (box_width / px_per_mm) / (local_glyph_mm * (1.0 + page.tracking)))
            marks = _layout_scribble_at(request, page_i, block_i)
            lines = _lines_for_text(spec.text, columns_em, spec.indent, marks)
            baselines = _box_baselines(box_top, box_bottom, local_pitch, page.forced_baseline_ys)
            if len(lines) > len(baselines):
                shrink = max(0.58, len(baselines) / max(1, len(lines)))
                local_glyph_mm = max(3.2, local_glyph_mm * shrink)
                local_em = local_glyph_mm * px_per_mm
                local_pitch = max(local_em * 1.15, local_pitch * shrink)
                columns_em = max(1.0, (box_width / px_per_mm) / (local_glyph_mm * (1.0 + page.tracking)))
                lines = _lines_for_text(spec.text, columns_em, spec.indent, marks)
                baselines = _box_baselines(box_top, box_bottom, local_pitch, page.forced_baseline_ys)
                lines = lines[: len(baselines)]
            for line_i, raw_line in enumerate(lines):
                if line_i >= len(baselines):
                    break
                line, line_align = _as_flow_line(raw_line)
                if line == [_PAGE_BREAK]:
                    continue
                baseline_y = baselines[line_i]
                align = line_align if line_align in ("center", "right") else spec.align
                pen_x = _line_start_x(line, box_left, box_width, local_em, page.tracking, align)
                for char, scratch in _flagged_items(raw_line):
                    origin = pen_x
                    step = _place_glyph(
                        char,
                        char_index=char_index,
                        total_chars=total_chars,
                        page_i=page_i,
                        pen_x=origin,
                        baseline_y=baseline_y,
                        em_px=local_em,
                        tracking=page.tracking,
                        placements=placements,
                        jobs=jobs,
                        occurrence_counter=occurrence_counter,
                        fallbacks=fallbacks,
                        missing=missing,
                        variant_budgets=variant_budgets,
                        variants_per_char=request.variants_per_char,
                        style_id=spec.style_id,
                    )
                    if scratch:
                        _add_scratch_placement(
                            page_i=page_i,
                            pen_x=origin,
                            baseline_y=baseline_y,
                            em_px=local_em,
                            index=min(char_index, total_chars - 1),
                            placements=placements,
                            jobs=jobs,
                            scribble_style=request.scribble_style,
                        )
                    pen_x += step
                    if pen_x > box_right + local_em:
                        break
                    char_index += 1
    return char_index


def _baseline_positions(
    page: PageSpec,
    paper: PaperSpec,
    px_per_mm: float,
    pitch_px: float,
    *,
    pitch_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """决定每一行的基线像素位置。

    有横线/网格时基线必须落在纸张的**真实线位**上，而不是自己按边距推算 —— 
    否则文字和横线只是"碰巧接近"，放大后必然错位。这是背景自适应的核心约束。

    空白纸上没有线可依，行距由手决定：逐行做 AR(1) 漂移，而不是等距推进。
    """
    top = page.margin_top_mm * px_per_mm
    bottom = (page.height_mm - page.margin_bottom_mm) * px_per_mm

    if page.forced_baseline_ys:
        # 检测报的是整张纸上的线，含页眉页脚那几条。裁到版心是排版的事，
        # 不该让检测去猜边距；不裁的话首行会压在纸的最上沿。
        inside = [float(y) for y in page.forced_baseline_ys if top <= y <= bottom]
        if inside:
            return inside
        # 版心与检出线不相交时，留在书写带内合成行，不要退回页眉页脚那些线。

    if page.bind_to_ruling and paper.ruling in ("lined", "grid"):
        rule_pitch = paper.line_pitch_mm * px_per_mm
        first = math.ceil(top / rule_pitch)
        positions = []
        index = first
        while index * rule_pitch <= bottom:
            positions.append(index * rule_pitch)
            index += 1
        if positions:
            return positions

    max_lines = max(1, int((bottom - top) / max(pitch_px * 0.7, 1.0)) + 2)
    if pitch_jitter > 1e-4 and rng is not None:
        wobble = _drift_series(max_lines, pitch_jitter, 0.55, 0.0, rng)
        steps = pitch_px * (1.0 + np.clip(wobble, -0.22, 0.22))
    else:
        steps = np.full(max_lines, pitch_px)

    positions: list[float] = []
    y = top + pitch_px
    for step in steps:
        if y > bottom:
            break
        positions.append(float(y))
        y += float(step)
    return positions or [top + pitch_px]


def _drifted_style(style: WriterStyle, slant_delta: float, pressure_delta: float) -> WriterStyle:
    structure = dataclasses.replace(style.structure, slant=style.structure.slant + slant_delta)
    brush = dataclasses.replace(
        style.brush, tiba_depth=max(0.0, style.brush.tiba_depth * (1.0 + pressure_delta))
    )
    return dataclasses.replace(style, structure=structure, brush=brush)


def _glyph_seed(seed: int, char: str, variant: int) -> int:
    payload = f"{seed}\x1f{char}\x1f{variant}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def _scale_stamp(mass, offset, scale):
    from PIL import Image
    height, width = mass.shape
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    img = Image.fromarray(mass, mode='F')
    scaled = np.array(img.resize((new_w, new_h), Image.Resampling.BILINEAR), dtype=np.float32, copy=True)
    return scaled, (int(round(offset[0] * scale)), int(round(offset[1] * scale)))


def _accumulate(target: np.ndarray, patch: np.ndarray, origin_x: int, origin_y: int) -> None:
    height, width = target.shape
    ph, pw = patch.shape
    x0, y0 = max(0, origin_x), max(0, origin_y)
    x1, y1 = min(width, origin_x + pw), min(height, origin_y + ph)
    if x1 <= x0 or y1 <= y0:
        return
    target[y0:y1, x0:x1] += patch[y0 - origin_y : y1 - origin_y, x0 - origin_x : x1 - origin_x]


__all__ = [
    "PageSpec",
    "RenderRequest",
    "RenderResult",
    "render",
    "srgb_to_linear",
    "variant_budget",
]
