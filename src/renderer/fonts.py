"""字体注册表（R1 Step 5）。

从 ``assets/fonts/manifest.json`` 加载字体清单，只暴露 ``status == "approved"``
且通过校验的字体。校验项（契约 §3）：

- ``schema_version`` 必须等于 ``SCHEMA_VERSION``；
- ``file`` 必须存在；
- ``sha256`` 若非空则必须匹配文件内容；
- ``commercial_use`` 必须为 ``true``；
- ``use`` 缺省视为 ``["style", "stamp"]``。``load_manifest()`` 只返回含 ``style``
  的条目，盖章专用字体走 ``load_stamp_fonts()``。

manifest 缺失 / 为空 / 非法时返回空列表，调用方（``core._resolve_font_for``）
回退到系统字体，**绝不崩溃**。
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FontSpec:
    """一条通过校验的字体条目。"""

    id: str
    display_name: str
    file: Path
    sha256: str
    commercial_use: bool


def _repo_root() -> Path:
    """向上查找仓库根（含 ``assets`` 目录的那一级）。"""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "assets" / "fonts").is_dir():
            return parent
    # 兜底：editable 布局 services/renderer/src/renderer -> 上四级即仓库根
    for parent in here.parents:
        if (parent / "assets").is_dir():
            return parent
    return here.parents[0]


def manifest_path() -> Path:
    """返回 manifest.json 的绝对路径。"""
    return _repo_root() / "assets" / "fonts" / "manifest.json"


def _roles(entry: dict) -> frozenset[str]:
    """缺省无 use 视为风格+盖章都可用。只认 style / stamp。"""
    raw = entry.get("use")
    if not isinstance(raw, list) or not raw:
        return frozenset({"style", "stamp"})
    found = frozenset(str(item) for item in raw if item in ("style", "stamp"))
    return found or frozenset({"style", "stamp"})


def _sha256_matches(path: Path, expected: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().lower() == expected.lower()


def _parse_entry(entry: object, *, role: Optional[str] = None) -> Optional[FontSpec]:
    """解析单条 manifest 条目；不通过校验则返回 None（跳过，不崩溃）。"""
    if not isinstance(entry, dict):
        return None

    font_id = entry.get("id")
    if not isinstance(font_id, str) or not font_id:
        return None
    if entry.get("status") != "approved":
        return None
    if entry.get("commercial_use") is not True:
        return None
    if role is not None and role not in _roles(entry):
        return None

    file_str = entry.get("file")
    if not isinstance(file_str, str) or not file_str:
        return None
    file_path = Path(file_str)
    if not file_path.is_absolute():
        file_path = _repo_root() / file_path
    if not file_path.exists():
        return None

    sha = entry.get("sha256") or ""
    if isinstance(sha, str) and sha and not _sha256_matches(file_path, sha):
        return None

    return FontSpec(
        id=font_id,
        display_name=entry.get("display_name") or font_id,
        file=file_path,
        sha256=sha if isinstance(sha, str) else "",
        commercial_use=True,
    )


def _load_fonts(path: Optional[Path], *, role: str) -> List[FontSpec]:
    p = Path(path) if path is not None else manifest_path()
    try:
        if not p.exists():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        return []
    fonts = data.get("fonts")
    if not isinstance(fonts, list):
        return []

    result: List[FontSpec] = []
    for entry in fonts:
        spec = _parse_entry(entry, role=role)
        if spec is not None:
            result.append(spec)
    return result


def load_manifest(path: Optional[Path] = None) -> List[FontSpec]:
    """加载风格字体。盖章专用条目（use 只有 stamp）不会出现。"""
    return _load_fonts(path, role="style")


def load_stamp_fonts(path: Optional[Path] = None) -> List[FontSpec]:
    """加载盖章字体。风格列表看不到这里多出来的 stamp-only 条目。"""
    return _load_fonts(path, role="stamp")


def get_approved_font(font_id: str) -> Optional[FontSpec]:
    """按 id 返回 approved 风格字体；不存在、未通过校验或 stamp-only 返回 None。"""
    for spec in load_manifest():
        if spec.id == font_id:
            return spec
    return None
