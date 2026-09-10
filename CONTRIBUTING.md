# 贡献指南

感谢你对 realistic-handwriting-engine 的关注！欢迎各种形式的贡献。

---

## 可以贡献什么？

- 🐛 **Bug 报告**：发现问题请提 Issue
- ✨ **功能建议**：有想法尽管说
- 📝 **文档改进**：修正错别字、补充说明
- 💻 **代码贡献**：修复 bug、优化性能、添加功能
- 🎨 **示例**：更多使用示例
- 🧪 **测试**：补充测试用例

---

## 开发环境搭建

```bash
# 克隆仓库
git clone https://github.com/yhe960809-ai/realistic-handwriting-engine.git
cd realistic-handwriting-engine

# 安装（开发模式）
pip install -e ".[dev]"

# 跑测试
python -m pytest tests/ -q

# 跑示例
python examples/render_demo.py
```

---

## 代码规范

- **Python 3.12+**
- 遵循 PEP 8
- 所有公共 API 需有文档字符串
- 新功能需附带测试
- 保持确定性：相同 seed 必须产生相同输出

---

## 提交 PR 的流程

1. Fork 本仓库
2. 创建你的特性分支（`git checkout -b feature/amazing-feature`）
3. 提交你的改动（`git commit -m 'Add some amazing feature'`）
4. 推送到分支（`git push origin feature/amazing-feature`）
5. 开启一个 Pull Request

---

## 注意事项

- 请确保所有测试通过
- 新增代码需保持与现有代码风格一致
- 涉及性能优化的 PR，请附上基准测试数据
- 涉及真实度变化的 PR，请附上指标对比

---

## 问答

有问题？开一个 Issue，我们会尽快回复。
