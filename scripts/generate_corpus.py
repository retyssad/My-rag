"""生成仿真的电商客服知识库语料（75 篇，4 种格式）。

结构：3 个类目（phone / laptop / earbuds） × 25 篇
  - 商品详情 (product_detail) × 8  ：含参数表（表格）
  - 选购对比 (comparison)     × 5  ：含多级标题与对比表
  - FAQ                       × 8  ：短问答，含同义改写
  - 售后政策 (policy)         × 4  ：长篇政策
另外：跨类目总则 (all) × 0，由 policy 中部分共享文档承担

输出：
  data/raw/{category}/{slug}.{ext}   （pdf / docx / html / md 混合）
  data/raw/manifest.json             （记录每篇的元信息）

运行：uv run python scripts/generate_corpus.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# ------------------------------------------------------------------ 数据定义

CATEGORIES = {
    "phone": "手机",
    "laptop": "笔记本电脑",
    "earbuds": "蓝牙耳机",
}

# 每类 8 款"虚构但合理"的产品（品牌名直接沿用真实品牌 + 虚构型号，避免与真实产品混淆）
PRODUCTS = {
    "phone": [
        ("星耀", "X30 Pro", 4999, "骁龙8 Gen3", "6.78英寸 2K 120Hz", "5000mAh", "120W"),
        ("星耀", "X30", 3499, "骁龙8 Gen2", "6.67英寸 1.5K 120Hz", "4800mAh", "67W"),
        ("星耀", "X30 Lite", 1999, "骁龙7 Gen3", "6.56英寸 1080P 90Hz", "5000mAh", "33W"),
        ("云雀", "Y9 Ultra", 5999, "天玑9300", "6.82英寸 2K 144Hz", "5400mAh", "100W"),
        ("云雀", "Y9", 3999, "天玑9200", "6.7英寸 1.5K 120Hz", "5000mAh", "80W"),
        ("云雀", "Y9 SE", 2799, "天玑8300", "6.6英寸 1080P 120Hz", "5000mAh", "67W"),
        ("极光", "A5 Pro", 4299, "骁龙8 Gen2", "6.7英寸 1.5K 120Hz", "5100mAh", "90W"),
        ("极光", "A5", 3299, "骁龙8+ Gen1", "6.6英寸 1080P 120Hz", "4800mAh", "66W"),
    ],
    "laptop": [
        ("星耀", "Book 14 Pro", 6999, "酷睿 Ultra 7", "14.2英寸 3K 120Hz", "70Wh", "1.39kg"),
        ("星耀", "Book 14", 5499, "酷睿 Ultra 5", "14英寸 2.8K 90Hz", "60Wh", "1.42kg"),
        ("云雀", "Air 13", 5999, "锐龙 7 8840U", "13.3英寸 2.5K 60Hz", "54Wh", "1.19kg"),
        ("云雀", "Air 15", 6499, "锐龙 7 8840U", "15.6英寸 2.5K 120Hz", "70Wh", "1.65kg"),
        ("极光", "Game 16", 9999, "酷睿 i9-14900HX", "16英寸 2.5K 240Hz", "99Wh", "2.45kg"),
        ("极光", "Game 16 Pro", 14999, "酷睿 i9-14900HX + RTX 4080", "16英寸 2.5K 240Hz", "99Wh", "2.6kg"),
        ("星耀", "Book 16", 6299, "酷睿 Ultra 5", "16英寸 2.5K 120Hz", "75Wh", "1.78kg"),
        ("云雀", "Air 14", 5699, "锐龙 5 8640U", "14英寸 2.2K 60Hz", "56Wh", "1.35kg"),
    ],
    "earbuds": [
        ("星耀", "Buds Air 3", 599, "蓝牙5.3", "38小时", "主动降噪 42dB", "IP54"),
        ("星耀", "Buds Air 3 Pro", 999, "蓝牙5.3", "42小时", "主动降噪 48dB", "IP55"),
        ("云雀", "Pods S", 399, "蓝牙5.2", "30小时", "主动降噪 35dB", "IPX4"),
        ("云雀", "Pods S Pro", 799, "蓝牙5.3", "40小时", "主动降噪 46dB", "IP55"),
        ("极光", "Sonic 2", 499, "蓝牙5.3", "36小时", "主动降噪 40dB", "IP54"),
        ("极光", "Sonic 2 Pro", 899, "蓝牙5.4", "45小时", "主动降噪 50dB", "IP55"),
        ("星耀", "Buds Lite", 199, "蓝牙5.2", "24小时", "无", "IPX4"),
        ("云雀", "Pods Lite", 159, "蓝牙5.2", "22小时", "无", "IPX4"),
    ],
}

# 售后政策主题（每个类目 4 篇）
POLICY_TOPICS = [
    ("退换货政策", "return"),
    ("保修与维修政策", "warranty"),
    ("配送与运费说明", "shipping"),
    ("发票与售后凭证说明", "invoice"),
]

FAQ_TEMPLATES = {
    "phone": [
        ("是否支持快充？", "本机支持有线快充，具体功率见商品详情页参数表。使用原装充电器与数据线可达到标称功率，第三方配件可能因协议不兼容而降速。"),
        ("支持无线充电吗？", "X30 Pro 与 Y9 Ultra 支持无线充电；其余型号不支持无线充电，请以商品详情页参数为准。"),
        ("电池续航怎么样？", "日常中度使用可满足一天续航，具体电池容量见参数表。重度游戏场景建议随身携带充电宝。"),
        ("可以用两张电话卡吗？", "支持双卡双待，卡槽为双 Nano-SIM，不支持同时插入存储卡。"),
        ("支持防水吗？", "整机支持生活防泼溅，不建议在游泳、淋浴等场景使用，进水不在保修范围内。"),
        ("系统是安卓吗？", "搭载基于 Android 定制的系统，预装应用可卸载，系统更新按机型分批次推送。"),
        ("支持 5G 吗？", "支持 5G 网络，实际速率取决于运营商网络覆盖与套餐。"),
        ("屏幕碎了保修吗？", "人为损坏（含跌落、挤压导致的屏幕碎裂）不在免费保修范围内，可付费维修。"),
    ],
    "laptop": [
        ("可以加内存吗？", "Book 系列内存为板载设计，不支持后期加装，请在购买时选择合适配置。Game 系列支持内存扩展。"),
        ("能玩大型游戏吗？", "Game 系列搭载独立显卡可流畅运行主流大型游戏；Book / Air 系列为轻薄本，适合办公与轻度娱乐。"),
        ("续航大概多久？", "轻薄本日常办公约 8~10 小时，游戏本高负载下约 2~3 小时，具体以实测为准。"),
        ("有雷电接口吗？", "Book 14 Pro、Game 16 系列配备雷电/USB4 接口，其余机型为 USB-C 3.2。"),
        ("可以自己加装硬盘吗？", "Game 系列支持加装 M.2 固态硬盘；Book / Air 系列部分机型支持，请咨询客服确认具体机型。"),
        ("散热怎么样？", "采用双风扇多热管设计，高负载下风扇噪音会上升，属于正常散热行为。"),
        ("预装正版系统吗？", "预装正版 Windows 系统，随机附带激活授权，重装可自动激活。"),
        ("适合设计/剪辑吗？", "视频剪辑建议选择带独立显卡的 Game 系列或高配 Book 系列；纯文字办公轻薄本即可。"),
    ],
    "earbuds": [
        ("降噪效果如何？", "Pro 系列支持主动降噪，降噪深度见商品详情页参数；非 Pro 系列无主动降噪功能。"),
        ("单次续航多久？", "单次充满可听歌约 6~8 小时，配合充电盒总续航见参数表。"),
        ("支持无线充电吗？", "Pro 系列充电盒支持无线充电，非 Pro 系列需使用有线充电。"),
        ("可以同时连两个设备吗？", "支持双设备连接，可在手机与电脑间切换，具体以机型和固件版本为准。"),
        ("防水吗？运动能戴吗？", "支持 IP54 及以上防尘防水，适合日常运动；不建议游泳时佩戴。"),
        ("延迟高吗？玩游戏可以吗？", "开启游戏模式后可降低延迟，适合日常手游；专业电竞需求建议使用有线耳机。"),
        ("丢了单只耳机怎么办？", "可单独购买补配单只耳机，需联系客服并支付配件费用，补配后需重新配对。"),
        ("兼容苹果手机吗？", "支持蓝牙连接，苹果设备可正常使用音频功能，部分 App 内弹窗等特色功能仅在自家生态可用。"),
    ],
}

COMPARISON_PAIRS = {
    "phone": [
        ("X30 Pro 与 X30 怎么选", "X30 Pro 与 X30"),
        ("Y9 Ultra 与 Y9 怎么选", "Y9 Ultra 与 Y9"),
        ("2000~3000 元手机选购指南", "X30 Lite / Y9 SE"),
        ("长续航手机怎么挑", "X30 Pro / Y9 Ultra"),
        ("拍照手机选购要点", "Y9 Ultra / X30 Pro"),
    ],
    "laptop": [
        ("轻薄本 14 寸与 16 寸怎么选", "Book 14 / Book 16"),
        ("游戏本选购指南", "Game 16 / Game 16 Pro"),
        ("办公本续航对比", "Book 14 / Air 13"),
        ("编程开发选什么笔记本", "Book 14 Pro / Game 16"),
        ("预算 6000 元笔记本怎么选", "Book 14 / Air 14"),
    ],
    "earbuds": [
        ("降噪耳机怎么选", "Buds Air 3 Pro / Sonic 2 Pro"),
        ("百元耳机选购指南", "Buds Lite / Pods Lite"),
        ("运动和通勤耳机区别", "Buds Air 3 / Pods S"),
        ("长续航耳机对比", "Sonic 2 Pro / Buds Air 3 Pro"),
        ("入门与旗舰耳机差异", "Pods Lite / Y9 系列"),
    ],
}


# ------------------------------------------------------------------ 内容构造

def product_detail_md(brand: str, model: str, price: int, chip: str, screen: str,
                      battery: str, extra: str, category: str) -> str:
    cat = CATEGORIES[category]
    lines = [
        f"# {brand} {model} 商品详情",
        "",
        f"**商品类目**：{cat}",
        f"**官方售价**：{price} 元",
        "",
        "## 产品简介",
        f"{brand} {model} 是一款面向主流用户的{cat}，"
        f"搭载 {chip}，配备 {screen}，官方定位为{'旗舰' if price > 4000 else '中端'}产品。",
        "",
        "## 核心参数",
        "",
        "| 参数项 | 规格 |",
        "| --- | --- |",
        f"| 处理器 | {chip} |",
        f"| 屏幕 | {screen} |",
        f"| 电池/续航 | {battery} |",
        f"| 特色参数 | {extra} |",
        f"| 售价 | {price} 元 |",
        "",
        "## 卖点",
        f"1. {chip} 提供流畅的日常使用体验；",
        f"2. {screen} 在同价位中表现突出；",
        f"3. {battery} 的续航/充电配置满足日常需求。",
        "",
        "## 包装清单",
        "主机 × 1、说明书 × 1、保修卡 × 1、原装线材 × 1。",
        "",
        "> 以上参数以官方公布为准，实际使用体验可能因环境而异。",
    ]
    return "\n".join(lines)


def comparison_md(category: str, title: str, subjects: str) -> str:
    cat = CATEGORIES[category]
    return "\n".join([
        f"# {title}",
        "",
        f"本文面向正在选购{cat}的用户，对比 {subjects}，给出选择建议。",
        "",
        "## 一、核心差异",
        f"{subjects} 的主要差异集中在性能、续航与价格三个维度。"
        "预算充足、追求性能的用户可优先考虑高配型号；注重性价比的用户建议选择中端型号。",
        "",
        "## 二、参数对比",
        "",
        "| 对比项 | 型号 A | 型号 B |",
        "| --- | --- | --- |",
        "| 性能定位 | 更强 | 够用 |",
        "| 价格 | 更高 | 更低 |",
        "| 适合人群 | 重度用户 | 普通用户 |",
        "",
        "## 三、选购建议",
        "### 3.1 如果你更看重性能",
        "选择高配型号，长期使用更从容。",
        "",
        "### 3.2 如果你更看重性价比",
        "选择中端型号即可满足日常需求，省下的预算可购买配件。",
        "",
        f"> 结论：没有绝对更好的{cat}，只有更适合你使用场景的选择。",
    ])


def faq_md(category: str, q: str, a: str) -> str:
    cat = CATEGORIES[category]
    return f"# {cat}常见问题：{q}\n\n**问题**：{q}\n\n**解答**：{a}\n"


def policy_md(category: str, topic: str, slug: str, brand_scope: str) -> str:
    cat = CATEGORIES[category]
    if slug == "return":
        body = (
            f"## 适用范围\n本政策适用于{cat}类商品。\n\n"
            "## 七天无理由退货\n自签收之日起 7 天内，商品保持完好、不影响二次销售的，可申请无理由退货。"
            "以下情形不适用：已激活且影响二次销售的数码产品、人为损坏、缺少配件。\n\n"
            "## 质量问题退换\n自签收之日起 15 天内出现非人为性能故障，可凭检测单申请换货或退货。\n\n"
            "## 退货流程\n1. 在订单页提交退货申请；2. 客服审核；3. 寄回商品；4. 验收后 3~5 个工作日退款。\n\n"
            "## 运费承担\n质量问题由商家承担运费；无理由退货由消费者承担寄回运费。"
        )
    elif slug == "warranty":
        body = (
            f"## 保修期限\n{cat}主机保修 12 个月，随机配件保修 3 个月（以购机发票为准）。\n\n"
            "## 保修范围\n非人为因素的硬件故障属于保修范围。\n\n"
            "## 不在保修范围\n1. 人为损坏（进水、跌落、私自拆机）；2. 非官方渠道维修；3. 不可抗力损坏。\n\n"
            "## 维修流程\n联系客服报修 → 提供购机凭证 → 寄修或送修 → 检测报价 → 维修 → 返还。\n\n"
            "## 延长保修\n可购买延保服务，具体以商品页说明为准。"
        )
    elif slug == "shipping":
        body = (
            "## 配送范围\n全国大部分地区支持配送，偏远地区可能延迟。\n\n"
            "## 发货时效\n现货商品在付款后 48 小时内发货。\n\n"
            "## 运费标准\n单笔订单满 99 元包邮，未满收取 10 元运费；偏远地区另行计费。\n\n"
            "## 物流查询\n发货后可在订单页查看物流单号。\n\n"
            "## 签收注意\n请当面验货，如有破损请拒收并联系客服。"
        )
    else:  # invoice
        body = (
            "## 发票类型\n支持开具电子普通发票与增值税专用发票。\n\n"
            "## 开票时间\n订单完成后可在订单页申请开票，电子发票一般 1~3 个工作日开具。\n\n"
            "## 售后凭证\n发票是保修与售后维权的重要凭证，请妥善保存。\n\n"
            "## 发票信息修改\n发票开具后如需修改抬头，请联系客服在当月内处理。"
        )
    return f"# {cat}·{topic}\n\n{body}\n"


# ------------------------------------------------------------------ 写出各格式

def parse_blocks(text: str) -> list[tuple]:
    """把简易 Markdown 文本解析成块：('h1'|'h2'|'h3'|'p'|'quote', str) 或 ('table', rows)。"""
    blocks: list[tuple] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("| "):
            table = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    table.append(cells)
                i += 1
            blocks.append(("table", table))
            continue
        if line.startswith("### "):
            blocks.append(("h3", line[4:]))
        elif line.startswith("## "):
            blocks.append(("h2", line[3:]))
        elif line.startswith("# "):
            blocks.append(("h1", line[2:]))
        elif line.startswith("> "):
            blocks.append(("quote", line[2:]))
        elif line.strip():
            blocks.append(("p", line.strip()))
        i += 1
    return blocks


def write_docx_from_md(path: Path, text: str) -> None:
    """写入 DOCX，表格用真实 docx 表格。"""
    from docx import Document

    doc = Document()
    for kind, payload in parse_blocks(text):
        if kind == "h1":
            doc.add_heading(payload, level=1)
        elif kind == "h2":
            doc.add_heading(payload, level=2)
        elif kind == "h3":
            doc.add_heading(payload, level=3)
        elif kind == "table":
            rows = payload
            if not rows:
                continue
            table = doc.add_table(rows=len(rows), cols=len(rows[0]))
            table.style = "Table Grid"
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    if c < len(table.rows[r].cells):
                        table.rows[r].cells[c].text = cell
        elif kind == "quote":
            p = doc.add_paragraph(payload)
            p.italic = True
        else:
            doc.add_paragraph(payload)
    doc.save(str(path))


def write_pdf_from_md(path: Path, text: str) -> None:
    """写入文字版 PDF；表格用 PyMuPDF 画出带边框的真实表格（可被 find_tables 识别）。"""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    margin, y = 50, 60
    page_w = page.rect.width

    def new_page_if_needed(need: float) -> None:
        nonlocal page, y
        if y + need > 780:
            page = doc.new_page()
            y = 60

    for kind, payload in parse_blocks(text):
        if kind == "table":
            rows = payload
            if not rows:
                continue
            n_rows, n_cols = len(rows), len(rows[0])
            col_w = (page_w - 2 * margin) / n_cols
            row_h = 22
            new_page_if_needed(row_h * n_rows + 10)
            top = y
            shape = page.new_shape()
            for r in range(n_rows + 1):
                shape.draw_line((margin, top + r * row_h), (margin + col_w * n_cols, top + r * row_h))
            for c in range(n_cols + 1):
                shape.draw_line((margin + c * col_w, top), (margin + c * col_w, top + n_rows * row_h))
            shape.finish(color=(0, 0, 0), width=0.7)
            shape.commit()
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    page.insert_text(
                        (margin + c * col_w + 4, top + r * row_h + 15),
                        cell, fontname="china-s", fontsize=9,
                    )
            y = top + n_rows * row_h + 12
            continue

        size = 16 if kind == "h1" else 13 if kind == "h2" else 11
        line = payload
        while line:
            chunk, line = line[:40], line[40:]
            new_page_if_needed(size + 8)
            page.insert_text((margin, y), chunk, fontname="china-s", fontsize=size)
            y += size + 7
        y += 4
    doc.save(str(path))
    doc.close()


def write_md(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_html(path: Path, text: str) -> None:
    """把 Markdown 风格的简单文本转成带表格的 HTML。"""
    html_lines = ['<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>']
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("| "):
            # 收集整张表
            table = []
            while i < len(lines) and lines[i].startswith("|"):
                table.append(lines[i])
                i += 1
            html_lines.append("<table border='1'>")
            for row_idx, row in enumerate(table):
                cells = [c.strip() for c in row.strip("|").split("|")]
                if all(set(c) <= set("-: ") for c in cells):
                    continue  # 分隔行
                tag = "th" if row_idx == 0 else "td"
                html_lines.append(
                    "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
                )
            html_lines.append("</table>")
            continue
        elif line.strip():
            html_lines.append(f"<p>{line}</p>")
        i += 1
    html_lines.append("</body></html>")
    path.write_text("\n".join(html_lines), encoding="utf-8")


# ------------------------------------------------------------------ 主流程

def build_manifest_entry(doc_id: int, category: str, doc_type: str, title: str,
                         path: Path) -> dict:
    return {
        "doc_id": doc_id,
        "title": title,
        "doc_type": doc_type,
        "product_category": category,
        "source_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "format": path.suffix.lstrip("."),
    }


def emit(cat_dir: Path, slug: str, fmt: str, md_text: str) -> Path:
    """按格式写出文档，四种格式统一从同一份 Markdown 文本派生（表格不丢失）。"""
    if fmt == "md":
        path = cat_dir / f"{slug}.md"
        write_md(path, md_text)
    elif fmt == "html":
        path = cat_dir / f"{slug}.html"
        write_html(path, md_text)
    elif fmt == "docx":
        path = cat_dir / f"{slug}.docx"
        write_docx_from_md(path, md_text)
    else:  # pdf
        path = cat_dir / f"{slug}.pdf"
        write_pdf_from_md(path, md_text)
    return path


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    doc_id = 0

    # 循环格式，让四种格式均匀分布
    formats = ["md", "pdf", "docx", "html"]

    for category, products in PRODUCTS.items():
        cat_dir = RAW / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        # 1) 商品详情 × 8（含参数表）
        for idx, (brand, model, price, chip, screen, battery, extra) in enumerate(products):
            doc_id += 1
            fmt = formats[idx % 4]
            title = f"{brand} {model} 商品详情"
            md = product_detail_md(brand, model, price, chip, screen, battery, extra, category)
            slug = f"detail_{brand}_{model}".replace(" ", "_").lower()
            path = emit(cat_dir, slug, fmt, md)
            manifest.append(build_manifest_entry(doc_id, category, "product_detail", title, path))

        # 2) 选购对比 × 5（含多级标题与对比表）
        for idx, (title, subjects) in enumerate(COMPARISON_PAIRS[category]):
            doc_id += 1
            fmt = formats[(idx + 1) % 4]
            md = comparison_md(category, title, subjects)
            slug = f"compare_{idx}_{title}".replace(" ", "_").replace("/", "")[:40].lower()
            path = emit(cat_dir, slug, fmt, md)
            manifest.append(build_manifest_entry(doc_id, category, "comparison", title, path))

        # 3) FAQ × 8
        for idx, (q, a) in enumerate(FAQ_TEMPLATES[category]):
            doc_id += 1
            fmt = formats[(idx + 2) % 4]
            md = faq_md(category, q, a)
            slug = f"faq_{idx}".lower()
            path = emit(cat_dir, slug, fmt, md)
            manifest.append(build_manifest_entry(
                doc_id, category, "faq", f"{CATEGORIES[category]}常见问题：{q}", path))

        # 4) 售后政策 × 4
        for idx, (topic, slugpart) in enumerate(POLICY_TOPICS):
            doc_id += 1
            fmt = formats[(idx + 3) % 4]
            md = policy_md(category, topic, slugpart, CATEGORIES[category])
            slug = f"policy_{slugpart}".lower()
            path = emit(cat_dir, slug, fmt, md)
            manifest.append(build_manifest_entry(
                doc_id, category, "policy", f"{CATEGORIES[category]}·{topic}", path))

    # 5) 跨类目总则（all）：全站售后总政策，制造跨文档/跨类目场景
    all_dir = RAW / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    for title, slug in [("全站售后总政策", "global_policy")]:
        doc_id += 1
        path = all_dir / f"{slug}.md"
        text = (
            "# 全站售后总政策\n\n"
            "## 总则\n本政策适用于本平台所有商品类目，各品类专属政策与之冲突时，以专属政策为准。\n\n"
            "## 统一退货时限\n所有商品自签收之日起 7 天内支持无理由退货（特殊说明除外）。\n\n"
            "## 统一保修凭证\n保修以购机发票或电子订单为准，请务必保留。\n\n"
            "## 争议处理\n如与商家就售后产生争议，可申请平台介入，平台将在 3 个工作日内响应。"
        )
        write_md(path, text)
        manifest.append(build_manifest_entry(doc_id, "all", "policy", title, path))

    (RAW / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 统计
    from collections import Counter

    by_fmt = Counter(m["format"] for m in manifest)
    by_cat = Counter(m["product_category"] for m in manifest)
    by_type = Counter(m["doc_type"] for m in manifest)
    print(f"生成 {len(manifest)} 篇文档 -> {RAW}")
    print(f"  按格式: {dict(by_fmt)}")
    print(f"  按类目: {dict(by_cat)}")
    print(f"  按类型: {dict(by_type)}")


if __name__ == "__main__":
    main()
