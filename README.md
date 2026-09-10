# Realistic Handwriting Engine

<p align="center">
  <em>生产级手写体合成引擎 — 从笔画骨架出发，沿物理运动轨迹书写</em>
  <br>
  <sub>来自「<strong>手写如真</strong>」微信小程序的生产内核 · <a href="#与手写如真完整版的区别">了解完整版 →</a></sub>
</p>

<p align="center">
  <a href="https://github.com/yhe960809-ai/realistic-handwriting-engine/actions/workflows/ci.yml">
    <img src="https://github.com/yhe960809-ai/realistic-handwriting-engine/actions/workflows/ci.yml/badge.svg" alt="CI Status">
  </a>
  <a href="https://pypi.org/project/realistic-handwriting-engine/">
    <img src="https://img.shields.io/pypi/v/realistic-handwriting-engine.svg" alt="PyPI version">
  </a>
  <a href="https://github.com/yhe960809-ai/realistic-handwriting-engine/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT">
  </a>
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/9,574%20CJK%20glyphs-8A2BE2" alt="9,574 CJK glyphs">
</p>

---

## 效果展示

> 以下为引擎直接输出，无后期处理。

| 笔记本横线纸（HD） | 白纸 + 方格纸 |
|:---:|:---:|
| [![笔记本横线纸](docs/images/demo-notebook-lined.jpg)](docs/images/demo-notebook-lined.jpg) | <img src="docs/images/demo-a4-plain.jpg" width="260"><br><img src="docs/images/demo-a4-grid.jpg" width="260"> |

**涂改模拟**

![涂改模拟](docs/images/demo-scribble.jpg)

---

## 这是什么？

不是字体填充，不是滤镜贴图。

**每一个汉字从 9,574 个真实笔画中线（skeleton）出发，**
沿书写轨迹（Σ-Lognormal 速度剖面 + 2/3 幂律）合成墨迹，
经纤维洇散、咖啡环效应、镜头散焦、传感器噪声等全链路成像仿真，
输出肉眼难以与真人书写区分的手写体图片。

引擎来自已上线的微信小程序「**手写如真**」，是从生产环境中抽出来的核心内核。

---

## 核心技术

### 骨架轨迹引擎（v3）

- **9,574 汉字骨架库**：每笔 24 采样点，11 种笔型，11 万+ 笔画（基于文鼎公众授权字型）
- **MLS 弹性形变**：每个字每次出现都有独立的弹性形变 —— "这一次写得瘦一点、右下角塌一点"，但仍然是合法的字
- **Σ-Lognormal 速度剖面**（Plamondon 快速人体运动理论）
- **2/3 幂律** v∝κ^(-1/3)：转折处自然加粗，收笔自然出锋
- **OU 过程生理噪声** + **AR(1) 漂移** + **线性疲劳**：长文书写的真实变化

### 墨迹物理

- **Hertz 接触** r∝P^(1/3)：压力 → 笔宽的物理模型
- **断墨 / 飞白**：有限储墨 + 恢复率 + 速度相关消耗
- **咖啡环效应**：干燥时墨向边界迁移，边缘比中心深
- **Beer-Lambert 分通道吸收**：薄墨处向墨水色相偏移，交叉处更黑
- **纸纤维洇散**：空间变化的吸水率，总墨量守恒

### 纸张 & 成像仿真

- **Value Noise 纤维纹理** + 色斑 + 横线 / 方格
- **光照场 + 暗角 + 纸张微曲**
- **镜头散焦 + 色差 + 传感器散粒噪声 + 读出噪声**
- **ISP 锐化 + JPEG 压缩**

### 其他特性

- PNG / JPG / 多页 PDF 输出
- 手动分页 (`---`)、右对齐 (`>>>`)、多块版式
- 涂改 / 错字模拟（单线、双线、叉号）
- **真实度判别器**（`metrics/`）：可量化合成与真人书写的差异
- **AIGC 内容标识**：内置 GB 45438-2025 合成内容显式/隐式标识
- 290+ 单元测试

---

## 快速开始

### 安装

```bash
pip install realistic-handwriting-engine
```

或从源码安装：

```bash
git clone https://github.com/yhe960809-ai/realistic-handwriting-engine.git
cd realistic-handwriting-engine
pip install -e .
```

> 需要 Python 3.12+。

### 基本用法

```python
from renderer import RenderConfig, render

png_bytes = render(
    "长风破浪会有时，直挂云帆济沧海。",
    RenderConfig(
        engine="v3",       # "v3" = 生产引擎；"handright" = 字体基线引擎
        preset="a4",       # a4 / notebook / letter
        paper_style="lined",  # plain / lined / grid
        seed=42,           # 同 seed 同结果（确定性）
        mode="hd",         # preview = 快速预览；hd = 高清成品
    ),
)

with open("out.png", "wb") as f:
    f.write(png_bytes)
```

### 更多示例

```bash
python examples/render_demo.py        # 三种纸张各一张
python examples/different_styles.py   # 不同风格 + 同字不同形（seed 对比）
python examples/multi_page_pdf.py     # 多页 PDF 输出
python examples/scribble_demo.py      # 涂改模拟
```

---

## 架构

```
文本输入
    │
    ▼
┌───────────────────────────┐
│  排版（折行 / 避头尾 / 缩进） │
└─────────────┬─────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  v3 引擎：骨架 → 轨迹 → 墨迹               │
│  ・skeleton     骨架库加载                │
│  ・trajectory   轨迹合成（MLS + 运动学）   │
│  ・ink          墨迹沉积（物理模型）       │
└─────────────┬───────────────────────────┘
              │
              ▼
┌───────────────────────────┐
│  纸面生成 → 洇散 → 合成      │
│  （纤维 / 横线 / 方格）       │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  成像仿真                    │
│  （光照/散焦/噪声/JPEG）    │
└─────────────┬─────────────┘
              │
              ▼
     PNG / JPG / PDF
```

更详细的技术原理见 [docs/architecture.md](docs/architecture.md)。

---

## 目录结构

```
realistic-handwriting-engine/
├── src/renderer/
│   ├── v3/                 # v3 生产引擎（核心）
│   │   ├── pipeline.py     # 渲染管线主入口
│   │   ├── trajectory.py   # 轨迹合成（MLS + Σ-Lognormal + 2/3 幂律）
│   │   ├── ink.py          # 墨迹沉积（Hertz 接触 / 断墨 / 咖啡环）
│   │   ├── paper.py        # 纸面生成 + 成像仿真
│   │   ├── skeleton.py     # 骨架库加载
│   │   ├── strokes.py      # 11 种笔型剖面
│   │   ├── style.py        # 书写者风格模型（WSM）
│   │   ├── filters.py      # 自实现滤波器
│   │   ├── mistakes.py     # 涂改 / 错字模拟
│   │   └── ...
│   ├── metrics/            # 真实度判别器与指标
│   │   ├── realism.py      # 图像统计指标
│   │   └── discriminator.py # 真伪判别器
│   ├── core.py             # RenderConfig / render() 入口
│   ├── engine.py           # 引擎抽象
│   ├── v3_engine.py        # v3 引擎适配层
│   ├── fonts.py            # 字体注册表（带 license 门禁）
│   └── aigc.py             # AIGC 内容标识
├── assets/
│   ├── skeleton-bank/      # 汉字骨架库（文鼎公众授权）
│   └── fonts/              # 字体清单（OFL，字体文件需自行下载）
├── tests/                  # 290+ 单元测试
├── examples/               # 可运行示例
└── docs/                   # 文档
```

---

## 与「手写如真」完整版的区别

本仓库是**开源的引擎内核**，来自微信小程序「手写如真」的生产代码。

| 功能 | 开源版 | 完整版（小程序） |
|---|:---:|:---:|
| v3 骨架轨迹渲染引擎 | ✅ | ✅ |
| 程序化纸张（白纸/横线/方格） | ✅ | ✅ |
| 真实度判别器与指标 | ✅ | ✅ |
| AIGC 内容标识 | ✅ | ✅ |
| 涂改 / 错字模拟 | ✅ | ✅ |
| 多页 PDF 输出 | ✅ | ✅ |
| **照片纸面检测与合成**（拍自己的纸） | ❌ | ✅ |
| **专属风格拟合**（上传样张复刻笔迹） | ❌ | ✅ |
| **横线检测与智能对齐** | ❌ | ✅ |
| 微信小程序前端 | ❌ | ✅ |
| 用户系统 / 支付 / 作品管理 | ❌ | ✅ |
| 管理后台 | ❌ | ✅ |
| Word / PDF 文档导入 | ❌ | ✅ |

**为什么不全开源？**
- 引擎核心开源：建立技术信任，吸引开发者，推动手写生成技术发展
- 产品层闭源：体验和运营是产品的护城河，也是可持续开发的经济基础
- 照片检测 / 风格拟合：这是第二技术壁垒，暂不公开

**想体验完整版？** 微信搜索「**手写如真**」小程序。

---

## 性能

| 场景 | 模式 | 耗时 |
|---|---|---|
| A4 预览（约 1 页） | v3 + preview | ~1.5s |
| A4 高清（约 1 页） | v3 + hd | ~4s |
| 1000 字 HD | v3 + hd | ~5s |

> 在普通笔记本 CPU 上测得。实际速度取决于文本长度、字号、纸张类型等。

---

## 测试

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

290+ 测试覆盖：骨架加载、轨迹合成、墨迹沉积、纸面生成、确定性验证、真实度指标、AIGC 标识等。

字体文件缺失时，依赖字体的测试会自动跳过（不影响核心引擎测试）。

---

## 常见问题

### Q: 跟字体有什么区别？

字体是"同一个字每次都一样"，v3 引擎是"同一个字每次写得不一样，但像同一个人写的"。
每个字都有独立的弹性形变（MLS）、不同的书写实例（速度/压力/起收笔细节），还有长文漂移和疲劳效应。

### Q: 支持哪些汉字？

9,574 个（基本区 + 扩展 A 区绝大部分），覆盖简体中文日常使用。
生僻字会回退到字体渲染（handright 基线引擎），保证不缺字。

### Q: 能生成我的字迹吗？

开源版不支持。完整版（小程序）的「专属风格」功能可以上传 1-3 张手写样张，自动测量并生成专属风格。

### Q: 可以商用吗？

代码本身是 MIT 协议，可自由商用。
但请注意：
- **骨架库**是文鼎公众授权，商用需遵守其条款
- **字体**是 OFL 协议，各字体有各自的商用授权状态（manifest 中标注了 `commercial_use`）
- **输出图片**的版权归你，但如果用于欺骗/伪造等非法用途，责任自负

### Q: 为什么仓库里没有字体文件？

字体文件体积大（每个几 MB 到十几 MB），而且部分字体的 redistribution 条款需要单独确认。
开源版只放了 `manifest.json`（字体清单 + sha256 校验 + 授权信息），你可以按清单从各字体官网自行下载。

---

## 许可证

- **代码**：MIT License（见 [LICENSE](LICENSE)）
- **骨架数据**：Arphic Public License（文鼎公众授权书），见 `assets/skeleton-bank/`
- **字体**：SIL OFL 1.1，字体文件需自行从对应开源项目下载
- **第三方组件**：handrightbeta / reportlab / fonttools 等，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

---

## 致谢

- 骨架数据来源于 [hanzi-writer](https://github.com/chanind/hanzi-writer)（基于文鼎公众授权字型）
- 手写基线引擎参考了 [handright](https://github.com/Gsllchb/Handright)
- 开源字体来自各 OFL 字体项目（霞鹜文楷、悠哉字体、小赖字体、马善政、站酷系列 等）
- 技术灵感来自 Plamondon 的 Sigma-Lognormal 运动学理论

---

## 相关链接

- 「手写如真」微信小程序：微信搜索「手写如真」
- [GitHub Issues](https://github.com/yhe960809-ai/realistic-handwriting-engine/issues) — 反馈问题
- [docs/architecture.md](docs/architecture.md) — 技术原理详解
- [CHANGELOG.md](CHANGELOG.md) — 版本变更

---

<p align="center">
  <sub>Built with ❤️ by 手写如真团队</sub>
</p>
