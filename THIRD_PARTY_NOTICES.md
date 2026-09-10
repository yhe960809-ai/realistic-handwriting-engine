# 第三方许可与来源声明

## 1. 候选工程参考

### handwriting-web

- **来源**: https://github.com/liuweiqing147/handwriting-web
- **许可证**: MIT License
- **版权**: Copyright (c) 2025 Liuweiqing
- **用途**: 仅作为后端生成、分页、异步任务和 PDF 链路的工程参考。
- **使用范围**: 不复制其 Vue 前端、SQLite 任务队列、Sentry/GA/Clarity 监控、latest 镜像、Watchtower 配置及来源不明字体。

---

## 2. 渲染内核

### handrightbeta

- **版本**: 8.7.0
- **来源**: https://github.com/14790897/Handright
- **上游**: https://github.com/Gsllchb/Handright（官方原版 handright）
- **许可证**: BSD 3-Clause License ✅
- **版权**: Copyright (c) 2017-2023, Chenghui LI (Gsllchb)
- **状态**: 已审计，已安装，进入生产依赖
- **使用范围**: `services/renderer` 手写渲染核心
- **许可证副本**: `services/renderer/LICENSES/handrightbeta-LICENSE.txt`

> handrightbeta 是官方 handright 的 fork/修改版，版本号高于原版。根据 BSD 3-Clause 条款，允许商用、修改和再分发，但须保留原始版权声明。本项目已按要求保留 LICENSE 副本。

---

### reportlab

- **版本**: 5.0.0（依赖声明 `>=4.0.0`）
- **来源**: https://www.reportlab.com / https://pypi.org/project/reportlab/
- **许可证**: BSD License ✅（宽松开源许可证，允许商用）
- **版权**: Copyright (c) 2000-2025, ReportLab Inc.
- **状态**: 已审计，已安装，进入生产依赖
- **用途**: `services/renderer` 多页 PDF 输出（替代已移除的 PyMuPDF）

> 用于将渲染好的多页 PNG 无损封装为 PDF。原 PyMuPDF 因闭源商业许可证风险已被彻底移除。

---

### fonttools

- **版本**: 4.63.0（依赖声明 `>=4.0.0`）
- **来源**: https://github.com/fonttools/fonttools
- **许可证**: MIT License ✅
- **状态**: 已审计，已安装，进入生产依赖
- **用途**: `services/renderer` 读取字体 cmap 表，可靠检测缺字（glyph 覆盖判断）

> 纯 Python 实现，无编译依赖。用于替代不可靠的 `PIL.ImageFont.getmask` 缺字检测（getmask 对不存在字符会回退到 .notdef 方框而不抛异常，无法可靠判断）。

---

## 2.5 骨架数据（v3 引擎的字形来源）

### hanzi-writer-data

- **版本**: 2.0.1
- **来源**: https://registry.npmjs.org/hanzi-writer-data/-/hanzi-writer-data-2.0.1.tgz
- **上游**: https://github.com/skishore/makemeahanzi
- **许可证**: **ARPHIC PUBLIC LICENSE**（文鼎公众授权书）
- **版权**: Copyright (C) 1999 Arphic Technology Co., Ltd. 文鼎科技开发股份有限公司
- **状态**: 已审计，产物入库，进入生产依赖
- **用途**: `assets/skeleton-bank/cjk-hanziwriter-v1/` —— 9,574 个汉字的**笔画中线**
  （`medians`），供 v3 引擎合成书写轨迹
- **许可证副本**: `assets/skeleton-bank/cjk-hanziwriter-v1/ARPHICPL.TXT`
- **溯源清单**: `assets/skeleton-bank/cjk-hanziwriter-v1/manifest.json`（含来源 URL、
  许可证名、各产物文件 sha256）

> **ARPHIC PUBLIC LICENSE 的分发义务**：允许免费使用、复制与再分发（含商用），
> 但**必须随附本授权书全文**，且不得单独出售字型本身。本项目的履行方式：
>
> 1. 授权书原文 `ARPHICPL.TXT` 与数据产物同目录存放，随仓库与部署包一并分发；
> 2. 本声明文件列明来源、许可证与版权人；
> 3. 构建脚本 `tools/skeleton/build_bank.py` 只提取笔画中线坐标，**不修改、不再
>    分发原始字型文件（.ttf/.ttc）**，产物是坐标点列而非字型；
> 4. 产品不以出售字型为业务 —— 售卖的是渲染服务次数。
>
> 授权书要求「衍生作品须同样以本授权书发布」的部分只约束字型本身；本项目的
> 渲染引擎代码不构成字型衍生作品。如后续要单独发布 `skeleton-bank` 数据包，
> 必须连同 `ARPHICPL.TXT` 一起发布。

---

## 3. 字体资产

> 全部字体许可证：**SIL Open Font License 1.1**（允许商用、修改、再分发；修改版须改名，见各字体 `reserved_name`）。
> 许可证副本：`LICENSES/fonts/<font-id>/OFL.txt`；来源证据：`LICENSES/fonts/<font-id>/source.json`。
> 清单与校验：`assets/fonts/manifest.json`（后端 loader 校验 sha256 + `commercial_use` + `approved`）。
> 二进制：`assets/fonts/files/<font-id>.ttf`。

### 已批准（approved，进入生产清单）

| id | 展示名 | 风格 | 版本 | 上游 | reserved_name |
|---|---|---|---|---|---|
| lxgw-wenkai | 霞鹜文楷 | 清秀 | 1.522 | https://github.com/lxgw/LxgwWenKai | 是 |
| yozai | 悠哉字体 | 日常 | 0.868 | https://github.com/lxgw/yozai-font | 是 |
| kose-xiaolai | 小赖字体 | 随写 | 3.126 | https://github.com/lxgw/kose-font | 否 |
| mashanzheng | 马善政毛笔楷书 | 毛笔 | 2.003 | https://github.com/googlefonts/mashanzheng | 否 |

### 盖章专用（approved，use 只有 stamp，不进风格列表）

| id | 展示名 | 用途 | 版本 | 上游 |
|---|---|---|---|---|
| cef-cjk | 快去写作业 | 手写感补漏 | 2.28 | https://github.com/Partyb0ssishere/cef-fonts-cjk |
| plangothic-p1 | 遍黑体 P1 | 扩展区兜底 | 2.9.5792 | https://github.com/Fitzgerald-Porthmouth-Koenigsegg/Plangothic_Project |
| plangothic-p2 | 遍黑体 P2 | 扩展区兜底 | 2.9.5792 | https://github.com/Fitzgerald-Porthmouth-Koenigsegg/Plangothic_Project |

> 这三款只给缺骨架的汉字盖章。`load_manifest()` / `get_approved_font()` 看不到它们，小程序字体列表也不暴露。

### 待批准（pending，未进入生产清单）

| id | 展示名 | 风格 | 版本 | 上游 |
|---|---|---|---|---|
| liujianmaocao | 柳建茂草书 | 草书 | 1.003 | https://github.com/googlefonts/liujianmaocao |
| longcang | 龙藏体 | 手写 | 2.001 | https://github.com/googlefonts/longcang |
| zhimangxing | 智忙星 | 行书 | 2.001 | https://github.com/googlefonts/zhimangxing |
| zcool-kuaile | 站酷快乐体 | 活泼 | 2.001 | https://github.com/googlefonts/zcool-kuaile |
| zcool-qingke-huangyou | 站酷庆科黄油体 | 圆润 | 1.000 | https://github.com/googlefonts/zcool-qingke-huangyou |

### 纸张/Logo 资产

- **当前状态**: 无授权纸张/Logo 资产
- **上线要求**: 只有 `license_status=approved` 且有书面授权证据的资产才能进入生产清单
