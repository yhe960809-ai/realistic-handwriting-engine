"""V3 补丁回滚开关。

出锋顿笔与纸纤维吸水洇散会改 HD 像素。默认走补丁后行为。
回到补丁前像素（出锋参数空转、吸水率丢弃）时二选一：

- 环境变量 ``HANDWRITE_V3_LEGACY_INK=1``
- 把下面的 ``USE_LEGACY_INK`` 改成 ``True``

日常行楷的 slant / aspect / size_sigma 用素材整页标定后，回到标定前默认值时二选一：

- 环境变量 ``HANDWRITE_V3_LEGACY_STYLE=1``
- 把下面的 ``USE_LEGACY_STYLE`` 改成 ``True``

生僻汉字默认用已批准 OFL 字体盖章。整页恢复「生僻字直接失败」时二选一：

- 环境变量 ``HANDWRITE_V3_FONT_STAMP=0``
- 把下面的 ``USE_FONT_STAMP`` 改成 ``False``

产品级回滚仍是 ``engine=handright``，与这里无关。
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "TRUE", "yes"}
_FALSY = {"0", "false", "FALSE", "no", "off"}

USE_LEGACY_INK = os.environ.get("HANDWRITE_V3_LEGACY_INK", "") in _TRUTHY

USE_LEGACY_STYLE = os.environ.get("HANDWRITE_V3_LEGACY_STYLE", "") in _TRUTHY


def font_stamp_enabled() -> bool:
    """盖章开关现读环境变量，方便单测切换而不用 reload 模块。"""
    raw = os.environ.get("HANDWRITE_V3_FONT_STAMP", "1").strip()
    return raw not in _FALSY


USE_FONT_STAMP = font_stamp_enabled()
