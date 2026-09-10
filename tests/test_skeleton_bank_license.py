"""骨架库的授权义务由代码守住，不靠流程记得（开发原则 #9）。

ARPHIC PUBLIC LICENSE 允许商用，但要求分发时随附授权书全文。授权文件一旦
被误删或改名，构建应当立刻失败，而不是等到上线后被发现。
"""

import json
from pathlib import Path

import pytest

from renderer.v3.skeleton import load_bank


def _bank_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "assets" / "skeleton-bank" / "cjk-hanziwriter-v1"
        if candidate.is_dir():
            return candidate
    pytest.skip("skeleton bank not present in this checkout")


def test_license_text_ships_with_the_data():
    license_file = _bank_dir() / "ARPHICPL.TXT"
    assert license_file.is_file(), "ARPHIC 授权书必须与骨架数据同目录分发"
    text = license_file.read_text(encoding="utf-8", errors="replace")
    assert "ARPHIC PUBLIC LICENSE" in text.upper()


def test_manifest_records_source_and_license():
    manifest = json.loads((_bank_dir() / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["source"]
    assert source["license"] == "ARPHIC PUBLIC LICENSE"
    assert source["license_file"] == "ARPHICPL.TXT"
    assert source["registry"].startswith("https://")
    # 每个产物都要有 sha256，缺字/换库时能立刻查出来源不一致
    for meta in manifest["files"].values():
        assert len(meta["sha256"]) == 64


def test_notices_file_declares_arphic():
    here = Path(__file__).resolve()
    notices = next(
        (p / "THIRD_PARTY_NOTICES.md" for p in here.parents if (p / "THIRD_PARTY_NOTICES.md").is_file()),
        None,
    )
    assert notices is not None, "THIRD_PARTY_NOTICES.md 缺失"
    text = notices.read_text(encoding="utf-8")
    assert "ARPHIC PUBLIC LICENSE" in text
    assert "hanzi-writer-data" in text
    assert "ARPHICPL.TXT" in text


def test_bank_reports_the_declared_character_count():
    manifest = json.loads((_bank_dir() / "manifest.json").read_text(encoding="utf-8"))
    assert load_bank().character_count == manifest["character_count"]
