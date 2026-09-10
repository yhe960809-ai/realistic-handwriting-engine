import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FONT_FILES_DIR = REPO_ROOT / "assets" / "fonts" / "files"


def _fonts_available() -> bool:
    """检查字体文件是否存在（开源版可能只放了 manifest 不放二进制）。"""
    if not FONT_FILES_DIR.is_dir():
        return False
    ttfs = list(FONT_FILES_DIR.glob("*.ttf"))
    return len(ttfs) >= 1


def pytest_collection_modifyitems(config, items):
    """字体文件缺失时跳过需要字体的测试。"""
    if _fonts_available():
        return
    skip_marker = pytest.mark.skip(reason="字体文件未安装（开源版仅含 manifest）")
    for item in items:
        # 标记了 stamp / font_file 的测试跳过
        name = item.nodeid.lower()
        if any(kw in name for kw in ("font_stamp", "stamp", "rare_hanzi", "ext_c")):
            item.add_marker(skip_marker)
