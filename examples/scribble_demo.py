"""错字 / 涂改模拟示例。

运行: python examples/scribble_demo.py
输出: docs/images/scribble-demo.png
"""
from pathlib import Path

from renderer import RenderConfig, render


def main():
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    text = """这是一段有涂改的手写文字。
写错了划掉重写很正常，谁写字不涂改呢。
有时候画一条线，有时候画两条线。
还有的时候干脆叉掉。
这就是真实手写的样子。"""

    png_data = render(
        text,
        RenderConfig(
            engine="v3",
            preset="a4",
            paper_style="lined",
            seed=99,
            mode="hd",
            # 涂改参数
            scribble_count=3,      # 涂改次数
            scribble_style="cross", # 涂改样式: strike / double_strike / cross
        ),
    )
    out_path = out_dir / "scribble-demo.png"
    out_path.write_bytes(png_data)
    print(f"已生成: {out_path}")


if __name__ == "__main__":
    main()
