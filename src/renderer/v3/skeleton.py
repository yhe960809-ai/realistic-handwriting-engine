"""骨架库加载：把 9,574 个汉字的有序笔画中线读进内存。

坐标系统一为左上原点、y 向下、[0,1] 归一（见 tools/skeleton/build_bank.py）。
渲染期只读，不联网，不回退到网络。
"""

from __future__ import annotations

import functools
import json
import pathlib
from dataclasses import dataclass

import numpy as np

_DEFAULT_BANK_ID = "cjk-hanziwriter-v1"


def _assets_root() -> pathlib.Path:
    import os
    env = os.environ.get("HANDWRITE_ASSETS_DIR")
    if env:
        return pathlib.Path(env).resolve()
    # <repo>/src/renderer/v3/skeleton.py -> <repo>/assets
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "assets" / "skeleton-bank").is_dir():
            return parent / "assets"
    # fallback: 假设是 pip 安装的包，向上找
    return here.parents[3] / "assets"


@dataclass(frozen=True)
class GlyphSkeleton:
    """一个字的骨架。"""

    char: str
    # [笔数, 采样点数, 2]，归一化到 [0,1]
    points: np.ndarray
    # 每笔的笔型 id（见 tools/skeleton/stroke_types.py）
    type_ids: np.ndarray
    has_hook: np.ndarray
    lengths: np.ndarray
    ink_weights: np.ndarray
    bbox: tuple[float, float, float, float]
    # {"kind": "lr"|"tb"|"single", "split": float, "groups": [[笔序], [笔序]]}
    structure: dict

    @property
    def stroke_count(self) -> int:
        return int(self.points.shape[0])


class SkeletonBank:
    """骨架库。加载一次，全进程共享。"""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"骨架库缺失：{manifest_path}。请先运行 tools/skeleton/build_bank.py 构建。"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(root / "strokes.npz") as payload:
            self._points = payload["points"]
            self._type_id = payload["type_id"]
            self._has_hook = payload["has_hook"]
            self._length = payload["length"]
            self._ink_weight = payload["ink_weight"]
        self._index: dict[str, dict] = json.loads(
            (root / "index.json").read_text(encoding="utf-8")
        )

    @property
    def bank_id(self) -> str:
        return str(self.manifest["bank_id"])

    @property
    def character_count(self) -> int:
        return len(self._index)

    def __contains__(self, char: str) -> bool:
        return char in self._index

    @functools.lru_cache(maxsize=8192)
    def get(self, char: str) -> GlyphSkeleton | None:
        entry = self._index.get(char)
        if entry is None:
            return None
        start = entry["s"]
        stop = start + entry["n"]
        return GlyphSkeleton(
            char=char,
            points=self._points[start:stop],
            type_ids=self._type_id[start:stop],
            has_hook=self._has_hook[start:stop],
            lengths=self._length[start:stop],
            ink_weights=self._ink_weight[start:stop],
            bbox=tuple(entry["bbox"]),  # type: ignore[arg-type]
            structure=entry["structure"],
        )

    def coverage(self, text: str) -> tuple[int, int, list[str]]:
        """返回 (命中数, 需渲染字符数, 未命中字符列表)。

        空白与换行不计入分母：它们不需要骨架。
        """
        misses: list[str] = []
        total = 0
        for char in text:
            if char.isspace():
                continue
            total += 1
            if char not in self._index:
                misses.append(char)
        return total - len(misses), total, sorted(set(misses))


@functools.lru_cache(maxsize=4)
def load_bank(bank_id: str = _DEFAULT_BANK_ID) -> SkeletonBank:
    return SkeletonBank(_assets_root() / "skeleton-bank" / bank_id)
