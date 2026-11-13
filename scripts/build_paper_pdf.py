# ruff: noqa: E501
"""Typeset the report as PDF: Markdown -> designed HTML -> headless Chrome print.

Chrome does the text layout, so Arabic (right-to-left, joined, diacritized) is set correctly.
The builder adds what Markdown cannot say: a cover page with three headline figures, each
quoted bayt set as two facing hemistichs, system cards for the side-by-side example,
open/closed badges on the list of acceptor holes, numbered section heads and running feet.
The SVG charts are inlined so they use the page fonts.

Usage: uv run --with markdown python scripts/build_paper_pdf.py
       -> paper/PAPER.pdf, paper/PAPER.ar.pdf (and the intermediate .html files)
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import markdown

from arabic_poet.config import resolve_path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FONTS = ("https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&"
         "family=Aref+Ruqaa:wght@400;700&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=Inter:wght@400;500;600;700&"
         "family=Noto+Naskh+Arabic:wght@400;500;600;700&"
         "family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&"
         "family=JetBrains+Mono:wght@400&display=swap")

T = {
    "en": {
        "kicker": "Technical report · November 2026", "foot": "Form Without Meaning",
        "stats": [("{free} → {con}", "of hemistichs accepted by our acceptor: fine-tuned 7B model, free, then under the meter constraint"),
                  ("{lo} – {hi}", "accepted for three hosted general models given the prompt only, with no constraint"),
                  ("94.8% · 34.2%", "Shaer's poems in the requested meter by its classifier · their abyat that pass the scansion checker")],
        "open": "Open", "closed": "Closed", "abstract": "Abstract",
    },
    "ar": {
        "kicker": "تقرير تقني · نوفمبر 2026", "foot": "شكلٌ بلا معنى",
        "stats": [("من {free} إلى {con}", "من الأشطر يقبلها مدقّقنا: النموذج المدرَّب (7B) حرًّا، ثم مقيَّدًا بالوزن"),
                  ("{lo} – {hi}", "لثلاثة نماذج عامة مستضافة أُعطيت الطلب وحده، بلا أي قيد"),
                  ("94.8% · 34.2%", "قصائد Shaer في بحرها بحسب برنامج أصحابه · أبياتها التي تجتاز مدقّق الوزن")],
        "open": "مفتوح", "closed": "أُغلق", "abstract": "الخلاصة",
    },
}

CSS = """
@page { size: A4; margin: 22mm 20mm 23mm 20mm;
  @bottom-center { content: counter(page); font-family:'Inter',sans-serif; font-size: 8.5pt; color:#8a8880; }
  @top-right { content: "FOOT"; font-family: HEADFONT; font-size: 8pt; color:#a09d94; letter-spacing: .3pt; } }
@page :first { margin: 0; @bottom-center { content: ""; } @top-right { content: ""; } }
:root { --ink:#1c1c1a; --muted:#5f5d57; --teal:#0f5c63; --teal-2:#e6f0ef; --clay:#b5603a; --paper:#f8f1e4;
  --paper-2:#f3ecdc; --rule:#e0dbcf; --open:#b4532a; --open-bg:#fbeee6; --closed-bg:#e6f0ef; }
* { box-sizing: border-box; }
html { font-size: 10.5pt; }
body { margin:0; color:var(--ink); font-family:'Source Serif 4',Georgia,serif; line-height:1.58; text-align:justify;
  hyphens:auto; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
body.ar { font-family:'Amiri','Noto Naskh Arabic',serif; font-size: 13.2pt; line-height:1.86; hyphens:none; }
.sans { font-family:'Inter',sans-serif; } body.ar .sans, body.ar h2, body.ar h3, body.ar th, body.ar .card-h,
body.ar figcaption, body.ar .badge, body.ar .stat p { font-family:'IBM Plex Sans Arabic','Noto Naskh Arabic',sans-serif; }

/* ---- cover ---- */
.cover { height: 297mm; break-after: page; position: relative; background: var(--paper); overflow: hidden; }
.cover img { display:block; width:100%; height: 132mm; object-fit: cover; object-position: center 55%; }
.cover .in { padding: 11mm 22mm 0; }
.kicker { font-family:'Inter',sans-serif; font-size: 8.6pt; letter-spacing: 1.6pt; text-transform: uppercase; color: var(--clay); font-weight:600; margin:0 0 5mm; text-align:start; }
body.ar .kicker { font-family:'IBM Plex Sans Arabic',sans-serif; letter-spacing:0; font-size: 10.5pt; }
.cover h1 { font-family:'Source Serif 4',serif; font-size: 38pt; line-height:1.08; font-weight:600; color: var(--teal); margin:0 0 4mm; letter-spacing:-.6pt; text-align:start; }
body.ar .cover h1 { font-family:'Aref Ruqaa','Amiri',serif; font-size: 66pt; font-weight:700; letter-spacing:0; line-height:1.3; margin: 2mm 0 2mm; }
.cover .sub { font-size: 14.5pt; line-height:1.4; color: var(--ink); margin:0 0 6mm; text-align:start; max-width: 150mm; }
body.ar .cover .sub { font-family:'Amiri',serif; font-size: 18pt; line-height:1.7; }
.cover .by { font-family:'Inter',sans-serif; font-size: 10pt; color: var(--muted); margin:0; text-align:start; }
body.ar .cover .by { font-family:'IBM Plex Sans Arabic',sans-serif; font-size: 11.5pt; }
.stats { position:absolute; inset-inline: 22mm; bottom: 30mm; display:grid; grid-template-columns: repeat(3, 1fr); gap: 7mm; }
.stat { border-top: 1.6pt solid var(--teal); padding-top: 3.5mm; }
.stat b { display:block; font-family:'Inter',sans-serif; font-weight:600; font-size: 18.5pt; white-space: nowrap; color: var(--teal); line-height:1.15; direction:ltr; text-align:start; unicode-bidi: plaintext; font-variant-numeric: tabular-nums; }
body.ar .stat b { text-align: right; direction: rtl; font-family:'IBM Plex Sans Arabic','Inter',sans-serif; }
.stat p { font-family:'Inter',sans-serif; font-size: 8.4pt; line-height:1.45; color: var(--muted); margin: 2mm 0 0; text-align:start; hyphens:none; }
body.ar .stat p { font-size: 9.6pt; line-height:1.7; }
.hamd { position:absolute; inset-inline: 22mm; bottom: 12mm; margin:0; direction: rtl; text-align:center; font-family:'Amiri',serif; font-size: 14pt; color: var(--clay); }

/* ---- text ---- */
h2 { font-family:'Inter',sans-serif; font-size: 15pt; line-height:1.25; margin: 24pt 0 9pt; color: var(--teal); font-weight:600; text-align:start;
  break-after: avoid; display:flex; align-items: baseline; gap: 9pt; border-bottom: .8pt solid var(--rule); padding-bottom: 6pt; }
body.ar h2 { font-size: 16pt; font-weight:600; line-height:1.6; }
h2 .n { font-family:'Inter',sans-serif; font-size: 11pt; font-weight:700; color:#fff; background: var(--teal); border-radius: 50%; min-width: 19pt; height: 19pt;
  display:inline-flex; align-items:center; justify-content:center; transform: translateY(-1.5pt); }
h3 { font-family:'Inter',sans-serif; font-size: 11.2pt; margin: 16pt 0 5pt; font-weight:600; text-align:start; break-after: avoid; color: var(--ink); }
body.ar h3 { font-size: 12.4pt; }
p { margin: 0 0 8pt; orphans: 3; widows: 3; }
a { color: var(--teal); text-decoration: none; }
strong { font-weight: 600; } body.ar strong { font-weight: 700; }
code { font-family:'JetBrains Mono',Menlo,monospace; font-size:.8em; background:#f1eee6; padding:.5pt 3pt; border-radius:2.5pt; overflow-wrap:anywhere; direction:ltr; unicode-bidi:isolate; }
code.arx { font-family:'Amiri',serif; font-size:1em; direction:rtl; padding:0 4pt; }
ul, ol { margin: 0 0 9pt; padding-inline-start: 17pt; } li { margin-bottom: 3.5pt; }
p:has(> strong:first-child:last-child) { break-after: avoid; }
.abstract { background: var(--paper); border-radius: 5pt; padding: 13pt 17pt 7pt; margin: 0 0 6pt; border-inline-start: 3pt solid var(--clay); }
.abstract h2 { border:0; margin:0 0 6pt; padding:0; font-size: 9pt; text-transform:uppercase; letter-spacing:1.5pt; color: var(--clay); }
body.ar .abstract h2 { letter-spacing:0; font-size: 13pt; }
.pull { font-family:'Source Serif 4',serif; font-size: 15pt; line-height:1.45; color: var(--teal); text-align:center; margin: 15pt 9mm; padding: 11pt 0; border-block: .9pt solid var(--clay); hyphens:none; break-inside: avoid; }
body.ar .pull { font-family:'Aref Ruqaa','Amiri',serif; font-size: 22pt; line-height:1.9; font-weight:400; }

/* ---- tables ---- */
table { width:100%; border-collapse: collapse; margin: 10pt 0 13pt; font-family:'Inter',sans-serif; font-size: 8.1pt; line-height:1.38; text-align:start;
  break-inside: avoid; font-variant-numeric: tabular-nums; }
body.ar table { font-family:'IBM Plex Sans Arabic',sans-serif; font-size: 8.8pt; line-height:1.65; }
th { text-align:start; font-weight:600; color: var(--teal); background: var(--teal-2); padding: 5pt 6pt; vertical-align: bottom; }
th:first-child { border-start-start-radius: 4pt; } th:last-child { border-start-end-radius: 4pt; }
td { padding: 4.2pt 6pt; border-bottom: .5pt solid var(--rule); vertical-align: top; }
td:first-child { font-weight: 500; color: var(--ink); }
tr.hl td { background: var(--paper); font-weight: 600; }
table.num td:not(:first-child), table.num th:not(:first-child) { text-align: center; }
table.num td:not(:first-child) { white-space: nowrap; }

/* ---- verse ---- */
blockquote { margin: 10pt 0 12pt; padding: 10pt 12pt; background: var(--paper); border-radius: 5pt; break-inside: avoid; direction: rtl; }
.bayt { display:grid; grid-template-columns: 1fr 1fr; column-gap: 9mm; font-family:'Amiri',serif; font-size: 14.2pt; line-height: 2.1; hyphens:none; }
.bayt span { text-align:center; white-space: nowrap; }
.bayt + .bayt { border-top: .5pt dotted #d9cfb9; }
.card { border: .8pt solid var(--rule); border-radius: 6pt; overflow:hidden; margin: 9pt 0 12pt; break-inside: avoid; }
.card-h { font-family:'Inter',sans-serif; font-size: 8.6pt; font-weight:600; color:#fff; background: var(--teal); padding: 4.5pt 11pt; text-align:start; }
body.ar .card-h { font-size: 10pt; }
.card blockquote { margin:0; border-radius:0; }

/* ---- figures ---- */
figure { margin: 13pt 0 14pt; break-inside: avoid; text-align:center; }
figure svg { direction:ltr; width:100%; height:auto; border-radius: 5pt; font-family:'Inter',sans-serif; }
body.ar figure svg { font-family:'IBM Plex Sans Arabic','Inter',sans-serif; }
.nb { white-space: nowrap; }
figcaption { font-family:'Inter',sans-serif; font-size: 8.3pt; color: var(--muted); margin-top: 5pt; text-align:center; }
body.ar figcaption { font-size: 9.4pt; }
img.art { display:block; width:100%; height: 60mm; object-fit: cover; object-position: center 50%; border-radius: 5pt; margin: 4pt 0 12pt; break-inside: avoid; }

/* ---- acceptor holes ---- */
ol.holes { list-style:none; padding:0; counter-reset: h; }
ol.holes li { counter-increment: h; display:grid; grid-template-columns: 17pt 1fr auto; gap: 7pt; align-items: start; padding: 5.5pt 0; border-bottom: .5pt solid var(--rule); margin:0; break-inside: avoid; }
ol.holes li::before { content: counter(h); font-family:'Inter',sans-serif; font-weight:700; font-size: 9.5pt; color: var(--clay); padding-top: 1.5pt; }
.badge { font-family:'Inter',sans-serif; font-size: 7.4pt; font-weight:600; padding: 1.5pt 7pt; border-radius: 20pt; white-space:nowrap; margin-top: 2.5pt; }
body.ar .badge { font-size: 8.6pt; }
.badge.open { color: var(--open); background: var(--open-bg); } .badge.closed { color: var(--teal); background: var(--closed-bg); }
"""

PULLS = ("فلا القيد يصنع المعنى", "ضبطنا الشكل وأحكمنا ضبطه")


def fmt(v: float) -> str:
    return f"{v + 1e-9:.2f}"


def cover(lang: str, title: str, sub: str, by: str, hamd: str) -> str:
    res = json.loads(resolve_path("paper/results.json").read_text(encoding="utf-8"))["systems"]
    nums = {"free": fmt(res["ALLaM-7B FT, free"]["exact"]),
            "con": fmt(res["ALLaM-7B FT, constrained, final acceptor"]["exact"]),
            "lo": fmt(min(v["exact"] for k, v in res.items() if "prompt only" in k)),
            "hi": fmt(max(v["exact"] for k, v in res.items() if "prompt only" in k))}
    t = T[lang]
    stats = "".join(f'<div class="stat"><b>{a.format(**nums)}</b><p>{b}</p></div>' for a, b in t["stats"])
    return (f'<section class="cover"><img src="figures/art_cover.jpg" alt=""><div class="in">'
            f'<p class="kicker">{t["kicker"]}</p><h1>{title}</h1><p class="sub">{sub}</p>'
            f'<p class="by">{by}</p></div><div class="stats">{stats}</div>'
            f'<p class="hamd">{hamd}</p></section>')


def verse(m: re.Match) -> str:
    """A quoted bayt becomes two facing hemistichs, as in a printed diwan."""
    rows = []
    for para in re.findall(r"<p>(.*?)</p>", m.group(0), flags=re.S):
        halves = [h.strip() for h in re.split(r"<br\s*/?>|\n", para) if h.strip()]
        for i in range(0, len(halves), 2):
            pair = halves[i:i + 2]
            rows.append('<div class="bayt">' + "".join(f"<span>{h}</span>" for h in pair) + "</div>")
    return "<blockquote>" + "".join(rows) + "</blockquote>"


def build(src: str, lang: str) -> Path:
    md_path = resolve_path(src)
    text = md_path.read_text(encoding="utf-8")
    text = re.sub(r'^<div dir="rtl">\s*$|^</div>\s*$', "", text, flags=re.M)
    html = markdown.markdown(text, extensions=["tables", "sane_lists"])
    t = T[lang]

    # ---- cover, from the title block ----
    head, _, rest = html.partition("<h2>")
    title = re.search(r"<h1>(.*?)</h1>", head, flags=re.S).group(1)
    sub_m = re.search(r"<h3>(.*?)</h3>", head, flags=re.S)
    if sub_m:
        sub = sub_m.group(1)
    else:
        title, _, sub = title.partition(": ")
    paras = re.findall(r"<p>(.*?)</p>", head, flags=re.S)
    by = re.sub(r"\s+", " ", paras[0]).rstrip(".")
    hamd = next((re.sub(r"<[^>]+>", "", p).split("\n")[0] for p in paras if "الحمد لله" in p), "")
    first, sep, tail = rest.partition("<h2>")
    first = re.sub(r"^[^<]*</h2>", f'{t["abstract"]}</h2>', first, count=1)
    html = f'<div class="abstract"><h2>{first}</div>{sep}{tail}'

    # ---- figures and illustrations ----
    def image(m: re.Match) -> str:
        alt, path = m.group("alt"), m.group("src")
        if path.endswith(".svg"):
            svg = (md_path.parent / path).read_text(encoding="utf-8")
            svg = re.sub(r'font-family="[^"]*"', "", svg, count=1)
            return f"<figure>{svg}<figcaption>{alt}</figcaption></figure>"
        return f'<img class="art" src="{path}" alt="">'

    html = re.sub(r'<p><img alt="(?P<alt>[^"]*)" src="(?P<src>[^"]+)"\s*/?></p>', image, html)

    # ---- verse, system cards ----
    html = re.sub(r"<blockquote>.*?</blockquote>", verse, html, flags=re.S)
    html = re.sub(r"<p><strong>([^<]{3,90})</strong></p>\s*(<blockquote>.*?</blockquote>)",
                  r'<div class="card"><div class="card-h">\1</div>\2</div>', html, flags=re.S)
    html = re.sub(r"<code>([^<]*[ء-ي][^<]*)</code>", r'<code class="arx">\1</code>', html)

    # ---- the list of acceptor holes: open / closed badges ----
    def holes(m: re.Match) -> str:
        items = re.findall(r"<li>(.*?)</li>", m.group(0), flags=re.S)
        if len(items) != 9:
            return m.group(0)
        out = []
        for it in items:
            is_open = bool(re.search(r"\bOpen\b|مفتوح", it))
            it = re.sub(r"\s*(?:Open|Closed)\.\s*$|\s*(?:مفتوح|فأغلقناه|فأغلقنا الباب بالمعجم)\.\s*$", "", it.strip())
            cls = "open" if is_open else "closed"
            out.append(f'<li><span>{it}</span><span class="badge {cls}">{t[cls]}</span></li>')
        return '<ol class="holes">' + "".join(out) + "</ol>"

    html = re.sub(r"<ol>.*?</ol>", holes, html, flags=re.S)

    # ---- section numbers, tables, pull quotes ----
    html = re.sub(r"<h2>(\d+)\.\s*(.*?)</h2>", r'<h2><span class="n">\1</span><span>\2</span></h2>', html)
    html = re.sub(r"<table>(.*?)</table>",
                  lambda m: ('<table class="num">' if re.search(r"<td>\s*(?:0?\.\d|\d+(?:\.\d+)?%)", m.group(1))
                             else "<table>") + m.group(1) + "</table>", html, flags=re.S)
    html = re.sub(r"<tr>(?=\s*<td>[^<]*(?:gpt-6|claude-fable))", '<tr class="hl">', html)
    for key in PULLS:
        html = re.sub(rf"<p>({re.escape(key)}.*?)</p>", r'<p class="pull">\1</p>', html, flags=re.S)

    # model names never break across lines
    names = re.compile(r"(?<![\w/\"=.-])((?:ALLaM-7B(?:-Instruct)?|Jais-3-32B|gpt-6\.1-sol|gpt-6-astra|claude-fable-5-2|Qwen3-30B|Yehia-7B|GPT-4o))(?![\w\"])(?![^<]*>)")
    parts = re.split(r"(<figure>.*?</figure>)", html, flags=re.S)  # never inside an SVG
    html = "".join(x if x.startswith("<figure>") else names.sub(r'<span class="nb">\1</span>', x)
                   for x in parts)
    head_font = "'IBM Plex Sans Arabic', sans-serif" if lang == "ar" else "'Inter', sans-serif"
    css = CSS.replace("FOOT", t["foot"]).replace("HEADFONT", head_font)
    page = (f'<!doctype html><html lang="{lang}" dir="{"rtl" if lang == "ar" else "ltr"}"><head>'
            f'<meta charset="utf-8"><title>{md_path.stem}</title>'
            f'<link rel="stylesheet" href="{FONTS}"><style>{css}</style></head>'
            f'<body class="{lang}">{cover(lang, title, sub, by, hamd)}{html}</body></html>')
    out_html = md_path.with_suffix(".html")
    out_html.write_text(page, encoding="utf-8")
    out_pdf = md_path.with_suffix(".pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=25000", f"--print-to-pdf={out_pdf}",
                    out_html.as_uri()], check=True, capture_output=True)
    return out_pdf


if __name__ == "__main__":
    for src, lang in (("paper/PAPER.md", "en"), ("paper/PAPER.ar.md", "ar")):
        print(build(src, lang))
