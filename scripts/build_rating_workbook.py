"""Write the blind rating workbook that goes to human raters (Arabic, right-to-left).

Reads artifacts/blind_eval/combined.json (build_blind_eval.py). Each rater gets the same poems
under the same codes in a different shuffled order, so any prefix a rater completes is a random
sample. Sheet 1 explains the task and the scales; sheet 2 is the rating grid with 1-5 dropdowns.
The key that maps codes to systems is not in the workbook.

Usage: uv run --with openpyxl python scripts/build_rating_workbook.py [raters=2]
       -> artifacts/blind_eval/تقييم_القصائد_مقيّم_<n>.xlsx (+ .csv)
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

from arabic_poet.config import resolve_path

METERS = {"basit": "البسيط", "kamil": "الكامل", "khafif": "الخفيف", "mutadarik": "المتدارك",
          "mutaqarib": "المتقارب", "rajaz": "الرجز", "ramal": "الرمل", "sari": "السريع",
          "tawil": "الطويل", "wafir": "الوافر"}
SCALES = [
    ("الالتزام بالموضوع", "هل القصيدة تتكلم في الموضوع المطلوب؟",
     ["لا علاقة لها بالموضوع", "تلمّح إليه في بيت أو كلمة", "في الموضوع مع خروج ظاهر عنه",
      "في الموضوع في أكثر أبياتها", "في الموضوع من أولها إلى آخرها"]),
    ("الترابط بين الأبيات", "هل يبني كل بيت على ما قبله؟",
     ["أبيات لا رابط بينها", "رابط ضعيف بين بيتين أو ثلاثة", "مترابطة مع قفزات غير مفهومة",
      "مترابطة في أكثرها", "فكرة تنمو من بيت إلى بيت بلا حشو"]),
    ("سلامة اللغة", "النحو والصرف وصحة الكلمات والتشكيل",
     ["كلمات لا وجود لها أو أخطاء كثيرة تفسد الفهم", "أخطاء متكررة", "سليمة في الأغلب مع ركاكة",
      "سليمة فصيحة، هفوات قليلة", "سليمة جزلة دقيقة الاختيار"]),
    ("الجودة الشعرية", "الصورة والعاطفة والأثر: هل هذا شعر يُستحسن؟",
     ["كلام موزون لا شعر فيه", "صورة أو لفتة واحدة باهتة", "صور مألوفة مستقيمة",
      "صور حيّة وبعضها جديد", "شعر مؤثر تودّ أن تحفظه"]),
    ("الوزن", "إن كنت تعرف العروض أو تثق بأذنك. وإلا فاتركها فارغة",
     ["لا وزن مسموع", "أشطر موزونة متفرقة", "موزون في الأغلب مع كسور كثيرة",
      "على البحر المطلوب مع كسر أو كسرين", "موزون تمامًا على البحر المطلوب"]),
    ("القافية", "هل التزمت حرف القافية المطلوب، وهل كلمات القافية في موضعها؟",
     ["لا قافية", "قافية متذبذبة", "موحّدة في الأغلب مع تكرار أو كلمات محشورة",
      "موحّدة مع عيب أو عيبين", "موحّدة، كلماتها متنوعة وفي موضعها"]),
    ("التقدير العام", "حكمك على القصيدة كلها",
     ["غير صالحة", "ضعيفة جدًا", "مسودة تحتاج إعادة كتابة", "جيدة مع تعديلات قليلة",
      "صالحة للنشر كما هي"]),
]
HEAD = ["الكود", "الموضوع المطلوب", "البحر", "حرف القافية", "عدد الأبيات", "القصيدة",
        *[s[0] for s in SCALES], "ملاحظات"]
FONT = "Arial"
TEAL, YELLOW, SAND = "0F5C63", "FFF4B8", "F8F1E4"
THIN = Side(style="thin", color="D9D4C7")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def layout(poem: str) -> str:
    """One bayt per line: sadr, a gap, 'ajuz."""
    out = []
    for b in poem.split("\n\n"):
        h = [x.strip() for x in b.split("\n") if x.strip()]
        out.append("      ***      ".join(h))
    return "\n".join(out)


def instructions(ws, n: int, rater: int) -> None:
    ws.sheet_view.rightToLeft = True
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 46
    for c in "CDEFG":
        ws.column_dimensions[c].width = 34
    rows = [
        ("تقييم قصائد", "h"),
        (f"نسخة المقيّم رقم {rater}", "p"),
        ("اسم المقيّم (اكتبه في الخانة الصفراء):", "name"),
        ("", "p"),
        ("ما المطلوب", "h2"),
        (f"في ورقة «التقييم» {n} قصيدة. كل قصيدة كُتبت استجابةً لطلب: موضوع وبحر وحرف قافية وعدد "
         "أبيات. اقرأ الطلب ثم القصيدة، وأعطِ كل بند درجة من 1 إلى 5 في الخانات الصفراء.", "p"),
        ("لا يُذكر من كتب القصيدة، وهذا مقصود. قيّم ما أمامك فقط.", "p"),
        ("القصائد مرتّبة ترتيبًا عشوائيًّا. إن لم يتسع وقتك لها كلها فقيّم من أولها بالترتيب "
         "وتوقف حيث شئت؛ كل ما تقيّمه نافع.", "p"),
        ("العلامة «…» مكان شطر تعني أن الشطر لم يُكتب. احسبها عيبًا في القصيدة.", "p"),
        ("كل بيت في سطر: الصدر، ثم ***، ثم العجز.", "p"),
        ("عمود «الوزن» لمن يعرف العروض أو يثق بأذنه. إن لم تكن متأكدًا فاتركه فارغًا.", "p"),
        ("في «ملاحظات» اكتب ما تشاء: كلمة خاطئة، بيت أعجبك، سبب الدرجة.", "p"),
        ("لا تغيّر عمود «الكود» ولا ترتيب الصفوف. واحفظ الملف بصيغته (xlsx) وأعده كما هو.", "p"),
        ("", "p"),
        ("معنى الدرجات", "h2"),
    ]
    r = 1
    for text, kind in rows:
        c = ws.cell(row=r, column=1, value=text)
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=4)
        c.alignment = Alignment(horizontal="right", vertical="top", wrap_text=True,
                                readingOrder=2)
        c.font = Font(name=FONT, size={"h": 20, "h2": 14}.get(kind, 12),
                      bold=kind in ("h", "h2"), color=TEAL if kind in ("h", "h2") else "1C1C1A")
        ws.row_dimensions[r].height = {"h": 34, "h2": 26}.get(kind, 44 if len(text) > 110 else 23)
        if kind == "name":
            ws.unmerge_cells(start_row=r, start_column=1, end_row=r, end_column=4)
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2)
            box = ws.cell(row=r, column=3)
            ws.merge_cells(start_row=r, start_column=3, end_row=r, end_column=4)
            box.fill = PatternFill("solid", fgColor=YELLOW)
            box.border = BOX
        r += 1
    hdr = ["البند", "السؤال", "1", "2", "3", "4", "5"]
    for i, t in enumerate(hdr, 1):
        c = ws.cell(row=r, column=i, value=t)
        c.font = Font(name=FONT, size=12, bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor=TEAL)
        c.alignment = Alignment(horizontal="center", vertical="center", readingOrder=2)
        c.border = BOX
    r += 1
    for name, q, anchors in SCALES:
        for i, t in enumerate([name, q, *anchors], 1):
            c = ws.cell(row=r, column=i, value=t)
            c.font = Font(name=FONT, size=11, bold=i == 1)
            c.alignment = Alignment(horizontal="right", vertical="top", wrap_text=True,
                                    readingOrder=2)
            c.border = BOX
        ws.row_dimensions[r].height = 48
        r += 1
    r += 1
    ex = ws.cell(row=r, column=1, value="مثال على صف مكتمل (للتوضيح فقط، ليس من القصائد المطلوب "
                                        "تقييمها):")
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=7)
    ex.font = Font(name=FONT, size=12, bold=True, color=TEAL)
    ex.alignment = Alignment(horizontal="right", readingOrder=2)
    r += 1
    sample = ["الموضوع: الصبر. البحر: الكامل. القافية: ر", "قصيدة من أربعة أبيات في الصبر، سليمة "
              "اللغة، صورها مألوفة، وبيتها الثالث خرج إلى الغزل.",
              "الموضوع 4 · الترابط 3", "اللغة 5 · الجودة 3", "الوزن 4 · القافية 5", "العام 4",
              "ملاحظة: «البيت الثالث خارج الموضوع»"]
    for i, t in enumerate(sample, 1):
        c = ws.cell(row=r, column=i, value=t)
        c.font = Font(name=FONT, size=11, italic=True)
        c.alignment = Alignment(horizontal="right", vertical="top", wrap_text=True, readingOrder=2)
        c.fill = PatternFill("solid", fgColor=SAND)
        c.border = BOX
    ws.row_dimensions[r].height = 52


def grid(ws, rows: list[dict]) -> None:
    ws.sheet_view.rightToLeft = True
    widths = [11, 24, 11, 9, 8, 78, *[13] * len(SCALES), 40]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w
    for i, t in enumerate(HEAD, 1):
        c = ws.cell(row=1, column=i, value=t)
        c.font = Font(name=FONT, size=12, bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor=TEAL)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True,
                                readingOrder=2)
        c.border = BOX
    ws.row_dimensions[1].height = 40
    ws.freeze_panes = "G2"
    first, last = 7, 6 + len(SCALES)
    dv = DataValidation(type="whole", operator="between", formula1="1", formula2="5",
                        allow_blank=True, showErrorMessage=True)
    dv.errorTitle, dv.error = "درجة غير صحيحة", "اكتب رقمًا من 1 إلى 5، أو اترك الخانة فارغة."
    dv.promptTitle, dv.prompt, dv.showInputMessage = "الدرجة", "من 1 (الأضعف) إلى 5 (الأفضل)", True
    ws.add_data_validation(dv)
    for r, row in enumerate(rows, 2):
        vals = [row["code"], row["topic"], row["meter"], row["rhyme"], row["abyat"], row["poem"]]
        for i, v in enumerate(vals, 1):
            c = ws.cell(row=r, column=i, value=v)
            c.font = Font(name=FONT, size=15 if i == 6 else 11)
            c.alignment = Alignment(horizontal="center" if i != 2 else "right", vertical="center",
                                    wrap_text=True, readingOrder=2)
            c.border = BOX
        for i in range(first, last + 2):
            c = ws.cell(row=r, column=i)
            c.fill = PatternFill("solid", fgColor=YELLOW)
            c.font = Font(name=FONT, size=13 if i <= last else 11)
            c.alignment = Alignment(horizontal="center" if i <= last else "right",
                                    vertical="center", wrap_text=True, readingOrder=2)
            c.border = BOX
        ws.row_dimensions[r].height = max(30 * (row["poem"].count("\n") + 1), 60)
    col = lambda i: ws.cell(row=1, column=i).column_letter  # noqa: E731
    dv.add(f"{col(first)}2:{col(last)}{len(rows) + 1}")


def main() -> None:
    raters = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    out = resolve_path("artifacts/blind_eval")
    src = json.loads((out / "combined.json").read_text(encoding="utf-8"))
    base = []
    for it in src["items"]:
        r = it["request"]
        for name in src["modes"]:
            code = hashlib.sha256(f"{it['id']}/{name}/blind-v1".encode()).hexdigest()[:8]
            base.append({"code": code, "topic": r["topic"],
                         "meter": METERS.get(r["meter"], r["meter"]),
                         "rhyme": r.get("rhyme") or "غير محدد", "abyat": r["line_count"],
                         "poem": layout(it[name]["poem"])})
    for k in range(1, raters + 1):
        rows = base[:]
        random.Random(100 + k).shuffle(rows)
        wb = Workbook()
        ws1 = wb.active
        ws1.title = "التعليمات"
        instructions(ws1, len(rows), k)
        grid(wb.create_sheet("التقييم"), rows)
        path = out / f"تقييم_القصائد_مقيّم_{k}.xlsx"
        wb.save(path)
        with (out / f"تقييم_القصائد_مقيّم_{k}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEAD)
            w.writerows([[r["code"], r["topic"], r["meter"], r["rhyme"], r["abyat"], r["poem"],
                          *[""] * (len(SCALES) + 1)] for r in rows])
        print(path, len(rows))


if __name__ == "__main__":
    main()
