"""快速上手示例：生成一张 A4 横线纸手写体图片。

运行方法:
    pip install -e .
    python examples/render_demo.py

输出: docs/images/demo-a4-lined.png 等
"""
from pathlib import Path

from renderer import RenderConfig, render


def main():
    text = """长风破浪会有时，直挂云帆济沧海。

这是「手写如真」渲染引擎的开源版本。
每一个汉字都从笔画中线出发，
沿着书写轨迹合成墨迹，
再经过纤维洇散、咖啡环效应、传感器噪声等成像仿真，
输出肉眼难以与真人书写区分的手写体图片。

- 支持 9,574 个汉字
- 多种内置书写风格
- 白纸 / 横线 / 方格 三种纸张
- 支持涂改、错字模拟
- 输出 PNG / JPG / PDF
"""

    out_dir = Path(__file__).resolve().parent.parent / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # A4 横线纸
    png_data = render(
        text,
        RenderConfig(
            engine="v3",
            preset="a4",
            paper_style="lined",
            seed=42,
            mode="hd",
        ),
    )
    out_path = out_dir / "demo-a4-lined.png"
    out_path.write_bytes(png_data)
    print(f"已生成: {out_path} ({len(png_data)} bytes)")

    # A4 白纸
    png_data = render(
        "你好，手写！\n\nHello, Handwriting!",
        RenderConfig(
            engine="v3",
            preset="a4",
            paper_style="plain",
            seed=123,
            mode="hd",
        ),
    )
    out_path = out_dir / "demo-a4-plain.png"
    out_path.write_bytes(png_data)
    print(f"已生成: {out_path} ({len(png_data)} bytes)")

    # A4 方格纸
    png_data = render(
        "手写如真\n手写如真\n手写如真\n手写如真\n手写如真",
        RenderConfig(
            engine="v3",
            preset="a4",
            paper_style="grid",
            seed=7,
            mode="hd",
        ),
    )
    out_path = out_dir / "demo-a4-grid.png"
    out_path.write_bytes(png_data)
    print(f"已生成: {out_path} ({len(png_data)} bytes)")


if __name__ == "__main__":
    main()
