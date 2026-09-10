"""多页 PDF 输出示例。

运行: python examples/multi_page_pdf.py
输出: docs/images/multi-page-demo.pdf
"""
from pathlib import Path

from renderer import RenderConfig, render


def main():
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 生成长文本，触发自动分页
    paragraphs = [
        "第一章 启程",
        "",
        "天将降大任于是人也，必先苦其心志，劳其筋骨，饿其体肤，空乏其身，行拂乱其所为，所以动心忍性，曾益其所不能。",
        "",
        "人恒过，然后能改；困于心，衡于虑，而后作；征于色，发于声，而后喻。入则无法家拂士，出则无敌国外患者，国恒亡。",
        "",
        "然后知生于忧患而死于安乐也。",
        "",
        "---",
        "",
        "第二章 行路难",
        "",
        "金樽清酒斗十千，玉盘珍羞直万钱。",
        "停杯投箸不能食，拔剑四顾心茫然。",
        "欲渡黄河冰塞川，将登太行雪满山。",
        "闲来垂钓碧溪上，忽复乘舟梦日边。",
        "行路难，行路难，多歧路，今安在？",
        "长风破浪会有时，直挂云帆济沧海。",
        "",
        "---",
        "",
        "第三章 劝学",
        "",
        "君子曰：学不可以已。青，取之于蓝，而青于蓝；冰，水为之，而寒于水。木直中绳，輮以为轮，其曲中规。虽有槁暴，不复挺者，輮使之然也。故木受绳则直，金就砺则利，君子博学而日参省乎己，则知明而行无过矣。",
        "",
        "吾尝终日而思矣，不如须臾之所学也；吾尝跂而望矣，不如登高之博见也。登高而招，臂非加长也，而见者远；顺风而呼，声非加疾也，而闻者彰。假舆马者，非利足也，而致千里；假舟楫者，非能水也，而绝江河。君子生非异也，善假于物也。",
    ]

    text = "\n".join(paragraphs)

    # PDF 输出
    pdf_data = render(
        text,
        RenderConfig(
            engine="v3",
            preset="a4",
            paper_style="lined",
            seed=42,
            mode="hd",
            format="pdf",
        ),
    )
    out_path = out_dir / "multi-page-demo.pdf"
    out_path.write_bytes(pdf_data)
    print(f"已生成 PDF: {out_path}")


if __name__ == "__main__":
    main()
