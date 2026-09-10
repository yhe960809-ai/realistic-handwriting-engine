"""生成 README 展示用的 demo 图（小尺寸，快速加载）。

运行: python examples/generate_demo_images.py
输出: docs/images/demo-*.png
"""
from pathlib import Path

from renderer import RenderConfig, render


def main():
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    text_lined = """懂了努力，才懂成长

有些词，我们从小听到大，以为
早就懂了，直到摔过跟头、走过弯路
才咂摸出它真正的味道。对我来说，
努力就是这样一个词。

初中到高一，我一直是班里最
拼的那个。天不亮就爬起来背单词，
课间把自己钉在座位上刷题，放学
回家写到后半夜是家常便饭。我总
觉得，努力就得是这样。"""

    text_plain = "长风破浪会有时，\n直挂云帆济沧海。"

    text_grid = "手写如真\n手写如真\n手写如真\n手写如真\n手写如真\n手写如真"

    text_scribble = """这是一段有涂改的文字。
写错了划掉重写很正常。
有时候画一条线，
有时候叉掉。
这就是真实手写的样子。"""

    # 横线纸（主展示图，用 hd 质量）
    png = render(
        text_lined,
        RenderConfig(engine="v3", preset="notebook", paper_style="lined", seed=42, mode="hd"),
    )
    (out_dir / "demo-notebook-lined.png").write_bytes(png)
    print(f"横线笔记本: {len(png)} bytes")

    # 白纸（小图）
    png = render(
        text_plain,
        RenderConfig(engine="v3", preset="a4", paper_style="plain", seed=123, mode="preview"),
    )
    (out_dir / "demo-a4-plain.png").write_bytes(png)
    print(f"白纸: {len(png)} bytes")

    # 方格纸（小图）
    png = render(
        text_grid,
        RenderConfig(engine="v3", preset="a4", paper_style="grid", seed=7, mode="preview"),
    )
    (out_dir / "demo-a4-grid.png").write_bytes(png)
    print(f"方格纸: {len(png)} bytes")

    # 涂改（小图）
    png = render(
        text_scribble,
        RenderConfig(
            engine="v3",
            preset="a4",
            paper_style="lined",
            seed=99,
            mode="preview",
            scribble_count=2,
            scribble_style="cross",
        ),
    )
    (out_dir / "demo-scribble.png").write_bytes(png)
    print(f"涂改: {len(png)} bytes")


if __name__ == "__main__":
    main()
