from arabic_poet.poetry import arud_fsa as fsa

SADR = "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
AJUZ = "بِسِقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ"


def test_real_verse_is_complete_in_its_meter_only():
    assert fsa.complete(SADR, "tawil", "sadr")
    assert fsa.complete(AJUZ, "tawil", "ajuz")
    assert not fsa.complete(SADR, "kamil", "sadr")


def test_every_prefix_of_a_real_hemistich_is_feasible():
    for i in range(1, len(SADR) + 1):
        assert fsa.feasible(SADR[:i], "tawil", "sadr"), SADR[:i]


def test_broken_hemistich_is_rejected():
    broken = SADR.replace("مِنْ ", "")
    assert not fsa.complete(broken, "tawil", "sadr")
    assert not fsa.feasible("قِفَا نَبْكِ ذِكْرَى حَبِيبٍ وَمَنْزِلِ ", "tawil", "sadr")


def test_non_arabic_is_not_well_formed():
    assert not fsa.feasible("قِفَا 1", "tawil", "sadr")


def test_well_formed_rejects_malformed_diacritics():
    from arabic_poet.poetry import arud_fsa as f

    bad = ["وَفِي حِينِ وَدَّْاِّعِ", "دَثِ يِّبْهِ", "مُتَعَجِّبًٌّاً مِنْ", "شَوْقًاً لِلْقِيَا",
           "وَلَا جُنُونِىَ", "دَرَّاَ اَءُ", "حَسَنًبِ"]
    good = ["شَوْقًا لِلْقِيَا", "شَوْقاً لِلْقِيَا", "فَتًى مَضَى", "الْاِنتظار", "بِهُدًى",
            "أَشَدُّ مِنَ الرَّدَى"]
    assert not any(f.well_formed(t) for t in bad)
    assert all(f.well_formed(t) for t in good)


def test_unwritten_alif_is_scanned():
    from arabic_poet.poetry import arud_fsa as f

    assert f.complete("وَيَقُولُ نَمْ فَغَدًا إِلَهُكَ يَرْزُقُ", "kamil", "ajuz")


def test_rhyme_letter_is_not_buried_under_two_suffix_letters():
    from arabic_poet.generation.constrained import _rhyme_ok

    assert not _rhyme_ok("غَيْرَ نَفْسِكَ رَاعِيًا", "ع", set())
    assert _rhyme_ok("فَيَتْبَعُ", "ع", set())
    assert _rhyme_ok("طَامِعًا", "ع", set())
    assert _rhyme_ok("مَدَامِعُهَا", "ع", set())
