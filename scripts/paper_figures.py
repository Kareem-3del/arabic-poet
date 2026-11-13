# ruff: noqa: E501
"""Draw the report's charts as SVG files (English and Arabic labels) from paper/results.json and
the saved experiment files. Plain SVG, no plotting library: text is laid out by the viewer, so
Arabic shapes correctly.

Usage: uv run python scripts/paper_figures.py   -> paper/figures/fig_*_{en,ar}.svg
"""
from __future__ import annotations

import json
from html import escape

from arabic_poet.config import resolve_path

# Teal, terracotta, violet: the report's colours, checked with the dataviz palette validator
# (lightness band, chroma floor, colour-blind and normal-vision separation, contrast): all pass.
SERIES = ("#0a94a3", "#d0673a", "#7d66b3")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FONT = {"en": "'Inter', 'Helvetica Neue', Arial, sans-serif",
        "ar": "'IBM Plex Sans Arabic', 'Geeza Pro', 'Arial', sans-serif"}


def hbar(path: str, lang: str, title: str, cats: list[str], series: dict[str, list[float | None]],
         unit: str = "", vmax: float = 1.0, note: str = "") -> None:
    """Grouped horizontal bars: one group per category, one thin bar per series, value at the
    bar's end. Legend above the plot; a single x axis."""
    rtl = lang == "ar"
    # In right-to-left text, "start" is the right edge: anchor names flip with the direction.
    right_aligned = "start" if rtl else "end"
    left_aligned = "end" if rtl else "start"
    dirn = "rtl" if rtl else "ltr"
    names = list(series)
    bar, gap, pad = 13, 3, 16
    group = len(names) * (bar + gap) - gap + pad
    label_w, right = 250, 64
    w = 760
    top = 78 if len(names) > 1 else 52
    plot_w = w - label_w - right
    h = top + len(cats) * group + (46 if note else 26)
    x0 = label_w
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
         f'font-family="{FONT[lang]}" direction="ltr" role="img" aria-label="{escape(title)}">',
         f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>',
         f'<text x="{w - 12 if rtl else 12}" y="26" font-size="15" font-weight="600" fill="{INK}" '
         f'text-anchor="{right_aligned if rtl else left_aligned}" direction="{dirn}">'
         f'{escape(title)}</text>']
    if len(names) > 1:  # legend: swatch + name, in ink
        x = w - 12 if rtl else 12
        for i, n in enumerate(names):
            tw = (6.6 if rtl else 7.2) * len(n) + 40
            if rtl:
                o.append(f'<rect x="{x - 12}" y="42" width="12" height="12" rx="2" fill="{SERIES[i]}"/>')
                o.append(f'<text x="{x - 18}" y="52" font-size="12" fill="{MUTED}" text-anchor="start" '
                         f'direction="rtl">{escape(n)}</text>')
                x -= tw
            else:
                o.append(f'<rect x="{x}" y="42" width="12" height="12" rx="2" fill="{SERIES[i]}"/>')
                o.append(f'<text x="{x + 18}" y="52" font-size="12" fill="{MUTED}">{escape(n)}</text>')
                x += tw
    for t in (0, 0.25, 0.5, 0.75, 1.0):  # recessive grid
        gx = x0 + plot_w * t
        o.append(f'<line x1="{gx:.1f}" y1="{top - 6}" x2="{gx:.1f}" y2="{top + len(cats) * group - pad + 4}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{gx:.1f}" y="{top + len(cats) * group - pad + 20}" font-size="11" fill="{MUTED}" '
                 f'text-anchor="middle">{t * vmax:g}{unit}</text>')
    for ci, c in enumerate(cats):
        gy = top + ci * group
        mid = gy + (group - pad) / 2 + 4
        o.append(f'<text x="{x0 - 10}" y="{mid:.1f}" font-size="12.5" fill="{INK}" text-anchor="{right_aligned}" '
                 f'direction="{dirn}">{escape(c)}</text>')
        for si, n in enumerate(names):
            v = series[n][ci]
            y = gy + si * (bar + gap)
            if v is None:
                o.append(f'<text x="{x0 + 6}" y="{y + 10.5}" font-size="11" fill="{MUTED}">'
                         f'{"لا ينطبق" if rtl else "n/a"}</text>')
                continue
            bw = max(plot_w * v / vmax, 1.5)
            r = min(4, bw / 2)
            # rounded data end, square at the baseline
            o.append(f'<path d="M{x0},{y} h{bw - r:.1f} a{r},{r} 0 0 1 {r},{r} v{bar - 2 * r} '
                     f'a{r},{r} 0 0 1 -{r},{r} h-{bw - r:.1f} z" fill="{SERIES[si]}"/>')
            txt = f"{v:.3f}".rstrip("0").rstrip(".") if vmax <= 1 else f"{v:g}"
            o.append(f'<text x="{x0 + bw + 6:.1f}" y="{y + 10.5}" font-size="11.5" fill="{INK}">'
                     f'{txt}{unit}</text>')
    if note:
        o.append(f'<text x="{w - 12 if rtl else 12}" y="{h - 10}" font-size="11" fill="{MUTED}" '
                 f'text-anchor="{right_aligned if rtl else left_aligned}" direction="{dirn}">'
                 f'{escape(note)}</text>')
    o.append("</svg>")
    resolve_path(path).write_text("\n".join(o), encoding="utf-8")


METERS_AR = {"basit": "البسيط", "kamil": "الكامل", "khafif": "الخفيف", "mutadarik": "المتدارك",
             "mutaqarib": "المتقارب", "rajaz": "الرجز", "ramal": "الرمل", "sari": "السريع",
             "tawil": "الطويل", "wafir": "الوافر"}


# Systems in reading order, with the group that gives each its colour.
SYS = [
    ("ALLaM-7B FT, free", "ALLaM-7B", "free"),
    ("Jais-3-32B FT, free", "Jais-3-32B", "free"),
    ("ALLaM-7B FT, constrained, final acceptor", "ALLaM-7B", "con"),
    ("Jais-3-32B FT, constrained, final acceptor", "Jais-3-32B", "con"),
    ("gpt-6.1-sol, prompt only", "gpt-6.1-sol", "host"),
    ("gpt-6-astra, prompt only", "gpt-6-astra", "host"),
    ("claude-fable-5-2, prompt only", "claude-fable-5-2", "host"),
]
GROUP = {"free": (SERIES[2], {"en": "fine-tuned, free", "ar": "مدرَّب، حرّ"}),
         "con": (SERIES[0], {"en": "fine-tuned, constrained", "ar": "مدرَّب، مقيَّد"}),
         "host": (SERIES[1], {"en": "hosted, prompt only", "ar": "مستضاف، بالطلب وحده"})}


def panels(path: str, lang: str, title: str, rows: list[tuple[str, str]],
           measures: list[tuple[str, list[float | None]]], note: str = "", vmax: float = 1.0,
           unit: str = "") -> None:
    """Small multiples: one panel per measure, the same systems in the same order in each,
    one bar per system coloured by its group. Reads as 'who is high on what'."""
    rtl = lang == "ar"
    ra, dirn = ("start" if rtl else "end"), ("rtl" if rtl else "ltr")
    w, label_w, gap, row_h, bar = 760, 150, 14, 21, 12
    n = len(measures)
    dec = 1 if vmax == 5 else 0
    pw = (w - label_w - 16 - gap * (n - 1)) / n
    top = 96
    h = top + len(rows) * row_h + (44 if note else 24)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
         f'font-family="{FONT[lang]}" direction="ltr" role="img" aria-label="{escape(title)}">',
         f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>',
         f'<text x="{w - 12 if rtl else 12}" y="26" font-size="15" font-weight="600" fill="{INK}" '
         f'text-anchor="start" direction="{dirn}">{escape(title)}</text>']
    x = w - 12 if rtl else 12
    for g in dict.fromkeys(r[1] for r in rows):
        col, names = GROUP[g]
        tw = (6.4 if rtl else 6.9) * len(names[lang]) + 40
        if rtl:
            o.append(f'<rect x="{x - 12}" y="40" width="12" height="12" rx="2" fill="{col}"/>')
            o.append(f'<text x="{x - 18}" y="50" font-size="12" fill="{MUTED}" text-anchor="start" '
                     f'direction="rtl">{escape(names[lang])}</text>')
            x -= tw
        else:
            o.append(f'<rect x="{x}" y="40" width="12" height="12" rx="2" fill="{col}"/>')
            o.append(f'<text x="{x + 18}" y="50" font-size="12" fill="{MUTED}">{escape(names[lang])}</text>')
            x += tw
    for ri, (name, _) in enumerate(rows):
        y = top + ri * row_h
        arabic = any("\u0600" <= ch <= "\u06ff" for ch in name)
        o.append(f'<text x="{label_w - 8}" y="{y + 10.5}" font-size="12" fill="{INK}" '
                 f'text-anchor="{"start" if arabic else "end"}" '
                 f'direction="{"rtl" if arabic else "ltr"}">{escape(name)}</text>')
    for mi, (mname, vals) in enumerate(measures):
        x0 = label_w + mi * (pw + gap)
        o.append(f'<text x="{x0 + (pw - 44) / 2:.1f}" y="{top - 14}" font-size="12" font-weight="600" '
                 f'fill="{INK}" text-anchor="middle" direction="{dirn}">{escape(mname)}</text>')
        for t in (0, 0.5, 1.0):
            gx = x0 + (pw - 44) * t
            o.append(f'<line x1="{gx:.1f}" y1="{top - 5}" x2="{gx:.1f}" y2="{top + len(rows) * row_h - 4}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
        for ri, (_, g) in enumerate(rows):
            v = vals[ri]
            y = top + ri * row_h
            if v is None:
                continue
            bw = max((pw - 44) * v / vmax, 1.5)
            r = min(4, bw / 2)
            o.append(f'<path d="M{x0:.1f},{y} h{bw - r:.1f} a{r},{r} 0 0 1 {r},{r} v{bar - 2 * r} '
                     f'a{r},{r} 0 0 1 -{r},{r} h-{bw - r:.1f} z" fill="{GROUP[g][0]}"/>')
            txt = f"{v + 1e-9:.2f}" if vmax <= 1 else f"{v:.{dec}f}"
            o.append(f'<text x="{x0 + bw + 5:.1f}" y="{y + 10}" font-size="11" fill="{INK}">{txt}{unit}</text>')
    if note:
        o.append(f'<text x="{w - 12 if rtl else 12}" y="{h - 10}" font-size="11" fill="{MUTED}" '
                 f'text-anchor="start" direction="{dirn}">{escape(note)}</text>')
    o.append("</svg>")
    _ = ra
    resolve_path(path).write_text("\n".join(o), encoding="utf-8")


def heat(path: str, lang: str, title: str, rows: list[str], cols: list[str],
         grid: list[list[float]], note: str = "") -> None:
    """A table of shares shaded in one hue, light to dark; every cell carries its number."""
    rtl = lang == "ar"
    dirn = "rtl" if rtl else "ltr"
    w, label_w, top, ch = 760, 170, 78, 30
    cw = (w - label_w - 12) / len(cols)
    h = top + len(rows) * ch + (40 if note else 16)

    def shade(v: float) -> str:
        lo, hi = (0xEA, 0xF4, 0xF5), (0x0A, 0x5F, 0x69)
        t = min(max((v - 0.4) / 0.6, 0.0), 1.0)  # the scale starts at 0.4: nothing here is lower
        return "#" + "".join(f"{round(a + (b - a) * t):02x}" for a, b in zip(lo, hi, strict=True))

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
         f'font-family="{FONT[lang]}" direction="ltr" role="img" aria-label="{escape(title)}">',
         f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>',
         f'<text x="{w - 12 if rtl else 12}" y="26" font-size="15" font-weight="600" fill="{INK}" '
         f'text-anchor="start" direction="{dirn}">{escape(title)}</text>']
    for ci, c in enumerate(cols):
        o.append(f'<text x="{label_w + ci * cw + cw / 2:.1f}" y="{top - 10}" font-size="11.5" '
                 f'fill="{MUTED}" text-anchor="middle" direction="{dirn}">{escape(c)}</text>')
    for ri, rname in enumerate(rows):
        y = top + ri * ch
        o.append(f'<text x="{label_w - 8}" y="{y + ch / 2 + 4:.1f}" font-size="12" fill="{INK}" '
                 f'text-anchor="end">{escape(rname)}</text>')
        for ci, v in enumerate(grid[ri]):
            x = label_w + ci * cw
            o.append(f'<rect x="{x + 1:.1f}" y="{y + 1}" width="{cw - 2:.1f}" height="{ch - 2}" rx="3" '
                     f'fill="{shade(v)}"/>')
            o.append(f'<text x="{x + cw / 2:.1f}" y="{y + ch / 2 + 4:.1f}" font-size="11.5" '
                     f'fill="{"#ffffff" if v > 0.72 else INK}" text-anchor="middle">{v:.2f}</text>')
    if note:
        o.append(f'<text x="{w - 12 if rtl else 12}" y="{h - 10}" font-size="11" fill="{MUTED}" '
                 f'text-anchor="start" direction="{dirn}">{escape(note)}</text>')
    o.append("</svg>")
    resolve_path(path).write_text("\n".join(o), encoding="utf-8")


def main() -> None:
    res = json.loads(resolve_path("paper/results.json").read_text(encoding="utf-8"))["systems"]
    out = resolve_path("paper/figures")
    out.mkdir(parents=True, exist_ok=True)
    have = [(k, n, g) for k, n, g in SYS if k in res]
    rows = [(n, g) for _, n, g in have]
    m1 = {"en": ("Accepted by our acceptor", "Independent checker ≥ 0.90", "Requested rhyme letter"),
          "ar": ("يقبله مدقّقنا", "المدقّق المستقل ≥ 0.90", "حرف القافية المطلوب")}
    t1 = {"en": "Who keeps the form? Three measures, same 30 prompts",
          "ar": "من يحفظ الشكل؟ ثلاثة مقاييس، والطلبات الثلاثون نفسها"}
    n1 = {"en": "Share of requested hemistichs or abyat, averaged over poems. One sample per prompt.",
          "ar": "نسبة ما طُلب من الأشطر أو الأبيات، متوسط القصائد. مرة واحدة لكل طلب."}
    for lang in ("en", "ar"):
        panels(f"paper/figures/fig_main_{lang}.svg", lang, t1[lang], rows,
               [(m1[lang][0], [res[k]["exact"] for k, _, _ in have]),
                (m1[lang][1], [res[k]["catt_090"] for k, _, _ in have]),
                (m1[lang][2], [res[k]["rawi_strict"] for k, _, _ in have])], note=n1[lang])

    # by meter: every system that writes mostly accepted verse
    strong = [(k, n) for k, n, g in have if g != "free"]
    order_m = sorted(METERS_AR, key=lambda m: -sum(res[k]["exact_by_meter"][m] for k, _ in strong))
    t2 = {"en": "Accepted hemistichs by meter", "ar": "الأشطر المقبولة بحسب البحر"}
    n2 = {"en": "One to six prompts per meter: read the pattern, not the decimals.",
          "ar": "من طلب إلى ستة طلبات لكل بحر: اقرأ الاتجاه لا الكسور."}
    for lang in ("en", "ar"):
        suffix = {"con": {"en": ", constrained", "ar": " مقيَّدًا"}, "host": {"en": "", "ar": ""}}
        names = [n + suffix[g][lang] for k, n, g in have if g != "free"]
        heat(f"paper/figures/fig_meter_{lang}.svg", lang, t2[lang], names,
             [m if lang == "en" else METERS_AR[m] for m in order_m],
             [[res[k]["exact_by_meter"][m] for m in order_m] for k, _ in strong], note=n2[lang])

    # time per poem
    timed = [(k, n, g) for k, n, g in have if g != "free" and res[k].get("seconds")]
    t5 = {"en": "Seconds per poem", "ar": "ثوانٍ لكل قصيدة"}
    n5 = {"en": "ALLaM on a laptop (M4 Pro), Jais on one L40S, hosted models through their APIs.",
          "ar": "ALLaM على حاسوب محمول (M4 Pro)، و Jais على L40S واحدة، والمستضافة عبر واجهاتها."}
    mx = max(res[k]["seconds"] for k, _, _ in timed)
    for lang in ("en", "ar"):
        panels(f"paper/figures/fig_time_{lang}.svg", lang, t5[lang], [(n, g) for _, n, g in timed],
               [("", [res[k]["seconds"] for k, _, _ in timed])], note=n5[lang], vmax=mx * 1.05)

    # staying on topic, by a lexical proxy
    t6 = {"en": "Topic words that appear in the poem", "ar": "كلمات الموضوع التي وردت في القصيدة"}
    n6 = {"en": "Share of the request's topic words found in the poem. Shared words, not meaning.",
          "ar": "نسبة كلمات الموضوع المطلوب التي وردت في القصيدة. كلمات مشتركة، لا معنى."}
    for lang in ("en", "ar"):
        panels(f"paper/figures/fig_topic_{lang}.svg", lang, t6[lang], rows,
               [("", [res[k]["topic_words"] for k, _, _ in have])], note=n6[lang], vmax=0.5)

    # ablation: one constraint at a time (ALLaM, seed 0)
    ab = [("ALLaM-7B FT, free", {"en": "free", "ar": "حرّ"}, "free"),
          ("ablation: diacritics only", {"en": "diacritics only", "ar": "تشكيل سليم فقط"}, "con"),
          ("ablation: rhyme only", {"en": "rhyme only", "ar": "قافية فقط"}, "con"),
          ("ablation: vocabulary only", {"en": "vocabulary only", "ar": "معجم فقط"}, "con"),
          ("ablation: meter only", {"en": "meter only", "ar": "وزن فقط"}, "con"),
          ("ablation: meter + rhyme", {"en": "meter + rhyme", "ar": "وزن وقافية"}, "con"),
          ("ALLaM-7B FT, constrained, final acceptor", {"en": "all constraints",
                                                         "ar": "القيود كلها"}, "con")]
    if all(k in res for k, _, _ in ab):
        t7 = {"en": "One constraint at a time (ALLaM-7B, 30 prompts)",
              "ar": "قيد واحد في كل مرة (ALLaM-7B، 30 طلبًا)"}
        m7 = {"en": ("Accepted by our acceptor", "Requested rhyme letter", "Topic words in the poem"),
              "ar": ("يقبله مدقّقنا", "حرف القافية المطلوب", "كلمات الموضوع في القصيدة")}
        for lang in ("en", "ar"):
            panels(f"paper/figures/fig_ablation_{lang}.svg", lang, t7[lang],
                   [(n[lang], g) for _, n, g in ab],
                   [(m7[lang][0], [res[k]["exact"] for k, _, _ in ab]),
                    (m7[lang][1], [res[k]["rawi_strict"] for k, _, _ in ab]),
                    (m7[lang][2], [res[k]["topic_words"] for k, _, _ in ab])])

    # blind human ratings (rater B, the sheet's 1-5 scale)
    hp = resolve_path("paper/human_eval.json")
    if hp.is_file():
        he = json.loads(hp.read_text(encoding="utf-8"))["means"]["B"]
        hs = [("allam_free", "ALLaM-7B", "free"), ("allam_constrained", "ALLaM-7B", "con"),
              ("jais_constrained", "Jais-3-32B", "con"), ("gpt61sol", "gpt-6.1-sol", "host"),
              ("claudefable52", "claude-fable-5-2", "host")]
        t8 = {"en": "Blind ratings by a human reader (1 to 5)",
              "ar": "تقييم قارئ بشري لا يعرف من كتب (من 1 إلى 5)"}
        m8 = {"en": ("On topic", "Abyat connect", "Sound language", "Poetic quality"),
              "ar": ("في الموضوع", "ترابط الأبيات", "سلامة اللغة", "الجودة الشعرية")}
        n8 = {"en": "Rater B, 150 poems. Rater A, on a 0-10 scale, ranks the poems the same way (Spearman 0.90 to 0.94).",
              "ar": "المقيّم «ب»، 150 قصيدة. والمقيّم «أ»، بمقياس من 0 إلى 10، يرتّب القصائد الترتيب نفسه (ارتباط 0.90 إلى 0.94)."}
        for lang in ("en", "ar"):
            panels(f"paper/figures/fig_human_{lang}.svg", lang, t8[lang], [(n, g) for _, n, g in hs],
                   [(m8[lang][i], [he[k][d] for k, _, _ in hs])
                    for i, d in enumerate(("topic", "coherence", "language", "quality"))],
                   note=n8[lang], vmax=5)

    sh = json.loads(resolve_path("paper/data/shaer_published_strict.json").read_text(
        encoding="utf-8"))["results"]
    keys = ["shaer", "reference_real_poems", next(k for k in sh if k not in ("shaer", "reference_real_poems"))]
    t3 = {"en": "Shaer's published poems: meter classifier and scansion checker",
          "ar": "قصائد Shaer المنشورة: برنامج تسمية البحر ومدقّق الوزن"}
    c3 = {"en": ["Shaer", "Real reference poems", "Untuned base model"],
          "ar": ["Shaer", "القصائد الحقيقية", "النموذج الأساس قبل التدريب"]}
    s3 = {"en": ("Classifier: poems in the requested meter", "Checker: abyat ≥ 0.90",
                 "Checker: abyat ≥ 0.95"),
          "ar": ("برنامجهم: قصائد في البحر المطلوب", "المدقّق: أبيات ≥ 0.90", "المدقّق: أبيات ≥ 0.95")}
    n3 = {"en": "The first series counts poems, the others count abyat: different measures.",
          "ar": "السلسلة الأولى تعدّ قصائد والباقيتان تعدّان أبياتًا: مقاييس مختلفة."}
    for lang in ("en", "ar"):
        a, b, c = s3[lang]
        hbar(f"paper/figures/fig_shaer_{lang}.svg", lang, t3[lang], c3[lang],
             {a: [sh[keys[0]].get("classifier_hit"), None, sh[keys[2]].get("classifier_hit")],
              b: [sh[k]["abyat_090"] for k in keys], c: [sh[k]["abyat_095"] for k in keys]},
             note=n3[lang])

    t4 = {"en": "One candidate per hemistich against four (ALLaM, 12 prompts)",
          "ar": "صيغة واحدة للشطر مقابل أربع صيغ (ALLaM، 12 طلبًا)"}
    c4 = {"en": ["Accepted by our acceptor", "Independent checker ≥ 0.90",
                 "Independent checker ≥ 0.95"],
          "ar": ["يقبله مدقّقنا", "المدقّق المستقل ≥ 0.90", "المدقّق المستقل ≥ 0.95"]}
    s4 = {"en": ("1 candidate", "4 candidates + repetition rule"),
          "ar": ("صيغة واحدة", "أربع صيغ وقاعدة التكرار")}
    for lang in ("en", "ar"):
        hbar(f"paper/figures/fig_cands_{lang}.svg", lang, t4[lang], c4[lang],
             {s4[lang][0]: [0.936, 0.667, 0.386], s4[lang][1]: [0.979, 0.844, 0.545]})
    print("figures written to", out)


if __name__ == "__main__":
    main()
