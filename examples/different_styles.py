"""展示不同书写风格的效果。

运行: python examples/different_styles.py
输出: docs/images/style-*.png
"""
from pathlib import Path

from renderer import RenderConfig, render


def main():
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    text = "长风破浪会有时，直挂云帆济沧海。"

    # v3 默认风格（行楷）
    png = render(
        text,
        RenderConfig(engine="v3", preset="a4", paper_style="plain", seed=42, mode="preview"),
    )
    (out_dir / "style-v3-xingkai.png").write_bytes(png)
    print("行楷风格: style-v3-xingkai.png")

    # handright 基线（字体填充）
    png = render(
        text,
        RenderConfig(engine="handright", preset="a4", paper_style="plain", seed=42, mode="preview"),
    )
    (out_dir / "style-handright.png").write_bytes(png)
    print("handright 基线: style-handright.png")

    # 不同 seed 对比（同字不同形）
    for i, seed in enumerate([1, 2, 3]):
        png = render(
            "同一个字每次写得不一样",
            RenderConfig(engine="v3", preset="a4", paper_style="plain", seed=seed, mode="preview"),
        )
        (out_dir / f"variant-seed-{seed}.png").write_bytes(png)
        print(f"seed={seed}: variant-seed-{seed}.png")


if __name__ == "__main__":
    main()
