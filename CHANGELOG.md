# Changelog

所有重要变更都记录在这个文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/) 规范。

---

## [0.1.0] - 2026-09-10

### Added

- 首次公开发布
- v3 骨架轨迹渲染引擎（核心 16 模块）
  - 9,574 汉字骨架库
  - MLS 弹性形变
  - Σ-Lognormal 速度剖面 + 2/3 幂律
  - Hertz 接触压力模型 + 断墨 + 咖啡环效应
  - Beer-Lambert 分通道墨色合成
  - Value Noise 纸张纤维 + 成像仿真链
- 真实度判别器与指标（metrics/）
- 双引擎架构：v3（生产）+ handright（基线/回退）
- AIGC 内容标识（GB 45438-2025）
- 涂改 / 错字模拟
- PNG / JPG / 多页 PDF 输出
- 290+ 单元测试
- 示例脚本：基础渲染、多风格、多页 PDF、涂改模拟
- 完整文档：README + architecture.md
- GitHub Actions CI
