"""Prompt builders. Pure functions: request in, chat messages out (no model dependency).

The same builders are used to create fine-tuning examples (``data/prepare.py``) so that
training prompts and inference prompts stay identical.
"""

from __future__ import annotations

from arabic_poet.generation.schemas import PoemRequest
from arabic_poet.models.base import Message
from arabic_poet.poetry.meter import meter_arabic_name

SYSTEM_AR = (
    "أنت شاعر عربي متمكّن، عالم بالعروض والقوافي وأساليب كبار الشعراء. "
    "تكتب شعرًا أصيلًا غير منقول، متماسك المعنى، صادق العاطفة، حيّ الصورة، رصين اللغة. "
    "لا تنسخ أبياتًا معروفة لشعراء آخرين ولا تحشو الأبيات بكلمات لا تضيف معنى."
)

STYLE_AR = {
    "classical": "قصيدة عمودية موزونة مقفّاة على نهج الشعر العربي الأصيل",
    "tafeela": (
        "قصيدة تفعيلة (شعر حر) تقوم على تكرار تفعيلة واحدة دون التزام عدد ثابت منها في السطر"
    ),
    "prose_poem": "قصيدة نثر ذات إيقاع داخلي وصور مكثّفة",
}

VARIETY_AR = {
    "fusha": "بالعربية الفصحى",
    "egyptian": "بالعامية المصرية",
    "levantine": "بالعامية الشامية",
    "gulf": "باللهجة الخليجية (الشعر النبطي)",
    "iraqi": "بالعامية العراقية",
    "maghrebi": "بالدارجة المغاربية",
}


#: Layout example only (al-Mutanabbi, 10th c., public domain). Generations that reproduce it
#: are copies, not originality; keep it out of training targets.
FORMAT_EXAMPLE_BAYT = (
    "الخَيلُ وَاللَيلُ وَالبَيداءُ تَعرِفُني",
    "وَالسَيفُ وَالرُمحُ وَالقِرطاسُ وَالقَلَمُ",
)


def unit_name(style: str) -> str:
    return "الأبيات" if style == "classical" else "الأسطر"


#: Shaer's training prompt (verbatim system text and field names from Shaer-AI training rows,
#: checked 2026-11-09). Output: one hemistich per line, no blank lines, no separators.
SHAER_SYSTEM = (
    "أنت شاعر عربي تكتب الشعر العمودي الكلاسيكي.\n"
    "التزم بالبحر المحدد في كل شطر، واستلهم من الموضوع دون نقله حرفياً.\n"
    "أخرج الأبيات فقط دون مقدمة أو تعليق.\n"
    "التزم التزاماً صارماً بالبحر المطلوب، ولا تخرج عنه.\n"
)


def shaer_messages(req: PoemRequest) -> list[Message]:
    """Shaer's native prompt format, for a like-for-like comparison with the Shaer adapter."""
    meter = (
        meter_arabic_name(req.meter) if req.meter else "الطويل"
    )  # e.g. "الخفيف", as in Shaer data
    n = 2 * req.line_count
    user = (
        f"البحر الأساسي: {meter}\nالصيغة: تام\nعدد الأشطر المطلوب: {n}\nالموضوع: {req.topic}\n\n"
        f"اكتب {n} أشطارًا ملتزمة بصيغة تام من بحر {meter} دون أي شرح إضافي."
    )
    return [{"role": "system", "content": SHAER_SYSTEM}, {"role": "user", "content": user}]


def format_rule(req: PoemRequest, separator: str, version: str = "v3") -> str:
    """Output-layout instruction.
    v1: bootstrap prompt (bayt per line, " ... " between hemistichs).
    v2: + explicit bayt/shatr definition, numbered abyat, counts in both units.
    v3: one shatr per line, blank line between abyat (the layout the model and Shaer's
        training data already favour); counts in both units. Non-classical styles: as v2."""
    sep = separator.strip()
    example = sep.join(FORMAT_EXAMPLE_BAYT)
    if version == "v1":
        if req.style == "classical":
            layout = (
                f"اكتب كل بيت كاملًا (صدره وعجزه) في سطر واحد، وافصل بين الشطرين بـ «{sep}» "
                f"داخل السطر نفسه. مثال على الشكل فقط، فلا تقتبسه: {example}"
            )
        else:
            layout = "اكتب كل سطر شعري في سطر مستقل."
        return (
            "اكتب القصيدة وحدها: بلا عنوان ولا مقدمة ولا شرح ولا ترقيم ولا علامات تنسيق. " + layout
        )
    n = req.line_count
    if version == "v3" and req.style == "classical":
        a, b = FORMAT_EXAMPLE_BAYT
        return (
            "طريقة الكتابة:\n"
            "- البيت شطران: الصدر ثم العجز. اكتب كل شطر في سطر مستقل: الصدر في سطر، "
            "والعجز في السطر الذي يليه مباشرة.\n"
            "- اترك سطرًا فارغًا واحدًا بين كل بيت والذي يليه.\n"
            f"- المطلوب {n} من الأبيات بالضبط، أي {2 * n} من الأسطر (الأشطر) في {n} من المجموعات.\n"
            "- لا ترقيم ولا نقاط ولا عنوان ولا شرح؛ الأشطر فقط.\n"
            f"مثال على شكل بيت واحد فقط، فلا تقتبسه:\n{a}\n{b}"
        )
    if req.style == "classical":
        return (
            "طريقة الكتابة:\n"
            f"- البيت سطر واحد فيه شطران: الصدر ثم العجز، وبينهما «{sep}». "
            "لا تضع العجز في سطر جديد.\n"
            f"- اكتب {n} من الأبيات بالضبط، أي {n} من الأسطر و{2 * n} من الأشطر، "
            "ورقّم كل بيت في أوله (1. 2. 3. ...).\n"
            "- اكتب الأبيات فقط: بلا عنوان ولا مقدمة ولا شرح.\n"
            f"مثال على شكل بيت واحد فقط، فلا تقتبسه: 1. {example}"
        )
    return (
        "طريقة الكتابة:\n"
        f"- اكتب {n} من الأسطر الشعرية بالضبط، كل سطر في سطر مستقل، ورقّم كل سطر في أوله "
        "(1. 2. 3. ...).\n"
        "- اكتب الأسطر فقط: بلا عنوان ولا مقدمة ولا شرح، ولا تستعمل النقاط الثلاث."
    )


ERA_AR = {
    "jahili": "العصر الجاهلي",
    "early_islamic": "صدر الإسلام والعصر الأموي",
    "abbasid": "العصر العباسي ومن عاصرهم في الأندلس والمشرق",
    "late": "العصور المتأخرة (المملوكي والعثماني)",
    "modern": "عصر النهضة والعصر الحديث",
}


def era_from_death_year(year: int | None) -> str | None:
    """Coarse period from the poet's death year (CE). Andalusian poets fall under their
    contemporaries' period; the corpus has no separate regional label."""
    if year is None:
        return None
    if year < 610:
        return "jahili"
    if year < 750:
        return "early_islamic"
    if year < 1258:
        return "abbasid"
    return "late" if year < 1800 else "modern"


def constraints_ar(req: PoemRequest, version: str = "v2") -> str:
    lines = [
        f"- الموضوع: {req.topic}",
        f"- الشكل: {STYLE_AR[req.style]}، {VARIETY_AR[req.language]}",
    ]
    if req.era:
        lines.append(f"- العصر والأسلوب: على نهج شعراء {ERA_AR[req.era]} في اللغة والصور والأغراض")
    if req.meter:
        lines.append(f"- البحر: {meter_arabic_name(req.meter)}، والتزمه في كل الأبيات")
    elif req.style == "classical":
        lines.append("- البحر: اختر بحرًا خليليًا واحدًا يناسب الموضوع والتزمه في كل الأبيات")
    no_repeat = "، بكلمة قافية مختلفة في كل بيت (لا تكرر كلمة القافية)" if version != "v1" else ""
    if req.rhyme:
        lines.append(f"- القافية: رويّها «{req.rhyme}» في آخر كل بيت{no_repeat}")
    elif req.style == "classical":
        lines.append(f"- القافية: قافية واحدة موحّدة الروي في آخر كل بيت{no_repeat}")
    lines.append(f"- عدد {unit_name(req.style)}: {req.line_count} بالضبط")
    if req.mood:
        lines.append(f"- النبرة العاطفية: {req.mood}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- Strategy A


def direct_messages(req: PoemRequest, separator: str, version: str = "v2") -> list[Message]:
    if version == "shaer":
        return shaer_messages(req)
    rule = format_rule(req, separator, version)
    user = f"اكتب قصيدة بهذه المواصفات:\n{constraints_ar(req, version)}\n\n{rule}"
    return [{"role": "system", "content": SYSTEM_AR}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- Strategy B

ARABIC_PLAN_HEADINGS = (
    "الفكرة المحورية:",
    "المسار العاطفي:",
    "الصور الشعرية: (من ثلاث إلى خمس صور مبتكرة غير مستهلكة)",
    "المعجم: (كلمات مفتاحية تناسب الموضوع)",
    "الوزن والقافية: (البحر، وحرف الروي، وكلمات قافية مرشّحة لا تتكرر)",
    "توزيع الأبيات: (مضمون كل بيت أو كل مجموعة أبيات)",
)


def arabic_plan_messages(req: PoemRequest) -> list[Message]:
    headings = "\n".join(ARABIC_PLAN_HEADINGS)
    user = (
        f"سأطلب منك قصيدة بهذه المواصفات:\n{constraints_ar(req)}\n\n"
        f"قبل كتابتها، ضع خطة موجزة لها بالعربية تحت هذه العناوين بالترتيب:\n{headings}\n\n"
        "اكتب الخطة فقط، ولا تكتب القصيدة."
    )
    return [{"role": "system", "content": SYSTEM_AR}, {"role": "user", "content": user}]


def poem_from_plan_messages(
    req: PoemRequest, plan: str, separator: str, version: str = "v2"
) -> list[Message]:
    user = (
        f"اكتب قصيدة بهذه المواصفات:\n{constraints_ar(req, version)}\n\n"
        f"وهذه خطتها:\n{plan}\n\n"
        f"اكتب القصيدة الآن ملتزمًا بالخطة وبالمواصفات.\n{format_rule(req, separator, version)}"
    )
    return [{"role": "system", "content": SYSTEM_AR}, {"role": "user", "content": user}]


def content_plan_messages(req: PoemRequest) -> list[Message]:
    """A meaning-only plan: one short prose sentence per bayt, no verse, no prosody."""
    n = req.line_count
    user = (
        f"سأطلب قصيدة بهذه المواصفات:\n{constraints_ar(req)}\n\n"
        f"اكتب خطة معانٍ لها في {n} أسطر مرقّمة، سطر لكل بيت: ماذا يقول البيت في جملة نثرية "
        "قصيرة واضحة. اجعل الأفكار متدرجة من بداية إلى خاتمة، ولا تكرر فكرة، والتزم بالموضوع. "
        "لا تكتب شعرًا ولا قوافي، ولا تضف شيئًا قبل الخطة أو بعدها."
    )
    return [{"role": "system", "content": SYSTEM_AR}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- Strategy C


def semantic_plan_messages(req: PoemRequest, language: str = "en") -> list[Message]:
    if language == "en":
        user = (
            "You are planning an Arabic poem. The request, in Arabic:\n"
            f"{constraints_ar(req)}\n\n"
            "Write a concise SEMANTIC plan in English only (meaning, not wording), with "
            "these headings:\nCentral idea:\nEmotional arc:\n"
            "Key images and metaphors: (3-5, concrete and original, avoid cliches)\n"
            "Structure: (what each group of lines does)\n\nDo not write the poem."
        )
        system = "You are an expert in Arabic poetry and poetics."
    else:
        user = (
            f"سأطلب منك قصيدة بهذه المواصفات:\n{constraints_ar(req)}\n\n"
            "ضع خطة دلالية موجزة (المعنى لا الألفاظ) تحت هذه العناوين:\n"
            "الفكرة المحورية:\nالمسار العاطفي:\nالصور والاستعارات: (من ثلاث إلى خمس)\n"
            "البناء: (وظيفة كل مجموعة أبيات)\n\nلا تكتب القصيدة."
        )
        system = SYSTEM_AR
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def linguistic_plan_messages(req: PoemRequest, semantic_plan: str) -> list[Message]:
    user = (
        f"المطلوب قصيدة بهذه المواصفات:\n{constraints_ar(req)}\n\n"
        f"وهذه خطتها الدلالية:\n{semantic_plan}\n\n"
        "حوّل هذه الخطة إلى خطة لغوية عروضية بالعربية تحت هذه العناوين:\n"
        "البحر وتفعيلاته:\n"
        f"الروي وكلمات القافية: ({req.line_count} كلمات قافية مرشّحة مختلفة، واحدة لكل بيت)\n"
        "المعجم والأساليب: (ألفاظ فصيحة وأساليب بلاغية مناسبة)\n"
        "مطلع القصيدة: (فكرة البيت الأول)\n\n"
        "اكتب الخطة فقط، ولا تكتب القصيدة."
    )
    return [{"role": "system", "content": SYSTEM_AR}, {"role": "user", "content": user}]


def poem_from_hybrid_messages(
    req: PoemRequest, semantic_plan: str, linguistic_plan: str, separator: str, version: str = "v2"
) -> list[Message]:
    user = (
        f"اكتب قصيدة بهذه المواصفات:\n{constraints_ar(req, version)}\n\n"
        f"الخطة الدلالية:\n{semantic_plan}\n\n"
        f"الخطة اللغوية والعروضية:\n{linguistic_plan}\n\n"
        f"اكتب القصيدة الآن ملتزمًا بالخطتين وبالمواصفات.\n{format_rule(req, separator, version)}"
    )
    return [{"role": "system", "content": SYSTEM_AR}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- Full tashkeel

TASHKEEL_RULE = "- اكتب كل كلمة مشكولة تشكيلًا كاملًا (الحركات والسكون والشدة والتنوين)."
_EXAMPLE_PLAIN = "الخَيلُ وَاللَيلُ وَالبَيداءُ تَعرِفُني\nوَالسَيفُ وَالرُمحُ وَالقِرطاسُ وَالقَلَمُ"
_EXAMPLE_TASHKEEL = (
    "الْخَيْلُ وَاللَّيْلُ وَالْبَيْدَاءُ تَعْرِفُنِي\n"
    "وَالسَّيْفُ وَالرُّمْحُ وَالْقِرْطَاسُ وَالْقَلَمُ"
)


def with_tashkeel(user: str) -> str:
    """The v3 request, asking for fully diacritized verse (same edit as the tashkeel SFT set)."""
    user = user.replace("- لا ترقيم ولا نقاط", TASHKEEL_RULE + "\n- لا ترقيم ولا نقاط")
    return user.replace(_EXAMPLE_PLAIN, _EXAMPLE_TASHKEEL)


def tashkeel_messages(req: PoemRequest) -> list[Message]:
    msgs = direct_messages(req, " ... ", "v3")
    msgs[1] = {"role": "user", "content": with_tashkeel(msgs[1]["content"])}
    return msgs


# --------------------------------------------------------------------------- bayt plan (v3)

PLAN_HEAD = "خطة الأبيات (اكتب كل بيت بكلماته، واختمه بكلمة قافيته، بالترتيب):"


def bayt_plan_block(plan: list[tuple[str, list[str]]]) -> str:
    """One line per bayt: its rhyme word, chosen before the bayt, and a few content words."""
    rows = [
        f"{i}. القافية: {rhyme}" + (f" | الكلمات: {'، '.join(words)}" if words else "")
        for i, (rhyme, words) in enumerate(plan, 1)
    ]
    return PLAN_HEAD + "\n" + "\n".join(rows)


def with_bayt_plan(user: str, plan: list[tuple[str, list[str]]]) -> str:
    """Insert the plan between the request's constraints and its writing rules."""
    head, sep, tail = user.partition("\n\nطريقة الكتابة:")
    return f"{head}\n\n{bayt_plan_block(plan)}{sep}{tail}"


def planned_tashkeel_messages(req: PoemRequest, plan: list[tuple[str, list[str]]]) -> list[Message]:
    msgs = tashkeel_messages(req)
    msgs[1] = {"role": "user", "content": with_bayt_plan(msgs[1]["content"], plan)}
    return msgs
