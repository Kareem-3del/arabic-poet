from arabic_poet.poetry.lexicon import Lexicon, norm


def lex() -> Lexicon:
    return Lexicon(norm(w) for w in ["قلب", "صبر", "الهوى", "دمعة", "في"])


def test_known_with_diacritics_and_proclitic():
    lx = lex()
    assert lx.known("قَلْبٌ")
    assert lx.known("وَالْقَلْبِ")
    assert lx.known("بِصَبْرٍ")
    assert not lx.known("دَوْخَجَّهُ")


def test_feasible_open_and_final():
    lx = lex()
    assert lx.feasible("فِي الْقَلْ")  # unfinished word can still become القلب
    assert not lx.feasible("فِي الْقَلْ", final=True)
    assert not lx.feasible("فِي الْغُصْ")
    assert lx.feasible("فِي الْقَلْبِ ")
    assert not lx.feasible("فِي غصر ")
    assert lx.feasible("وَ")


def test_fragments_and_glued_suffix_rejected():
    lx = Lexicon(norm(w) for w in ["قلب", "ال", "ح", "فخارها"])
    assert not lx.feasible("الْقَلْبِ الْ ", final=True)
    assert not lx.feasible("قَلْبٌ ح", final=True)
    assert not lx.feasible("فَخَارِهْا ")
    assert not lx.feasible("فَخَارِهْا", final=True)
    assert lx.feasible("فَخَارِهَا", final=True)
