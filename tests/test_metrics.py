from evaluation.metrics import cer, field_accuracy, levenshtein


def test_levenshtein():
    assert levenshtein("31/12/2027", "31/12/2027") == 0
    assert levenshtein("3l/12/2027", "31/12/2027") == 1
    assert levenshtein("", "abc") == 3


def test_cer_and_accuracy():
    assert cer(["31/12/2O27", None], ["31/12/2027", "AB"]) == (1 + 2) / 12
    assert field_accuracy(["a", "b", None], ["a", "c", None]) == 2 / 3
