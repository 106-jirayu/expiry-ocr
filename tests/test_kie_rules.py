from datetime import date

import pytest

from app.extraction.date_parser import find_dates, normalize_year
from app.extraction.kie import extract_fields, fix_digits, unclaimed_dates
from app.models import DocumentStatus
from app.recognition.base import TextLine
from app.rules.confidence import apply_confidence
from app.rules.router import route
from app.rules.validators import validate

TODAY = date(2026, 10, 28)


def L(text, conf=0.99):
    return TextLine(text, conf, [])


def run(lines):
    lines = [L(t) if isinstance(t, str) else t for t in lines]
    fields = apply_confidence(extract_fields(lines, TODAY))
    rules = validate(fields, TODAY, extra_dates=unclaimed_dates(lines, fields, TODAY))
    return fields, rules, route(fields, rules, 0.90)


@pytest.mark.parametrize("text,iso,fmt", [
    ("EXP 31/12/2027", "2027-12-31", "DD/MM/YYYY"),
    ("EXP: 2027-12-31", "2027-12-31", "YYYY-MM-DD"),
    ("BB 31.12.27", "2027-12-31", "DD/MM/YY"),
    ("BEST BEFORE 31 DEC 2027", "2027-12-31", "DD MON YYYY"),
    ("USE BY 31DEC27", "2027-12-31", "DD MON YYYY"),
    ("EXP DEC 2027", "2027-12-31", "MON YYYY"),
    ("EXP 12/2027", "2027-12-31", "MM/YYYY"),
    ("EXP 311227", "2027-12-31", "DDMMYY"),
    ("หมดอายุ 31/12/2570", "2027-12-31", "DD/MM/YYYY"),
    ("ควรบริโภคก่อน 31 ธ.ค. 70", "2027-12-31", "DD MON YYYY"),
])
def test_exp_formats(text, iso, fmt):
    f, _, _ = run([text])
    assert f["exp_date"].value == iso
    assert f["exp_date"].date_format == fmt


def test_full_label_auto_approved():
    f, rules, d = run(["MFG: 01/06/2026", "EXP: 01/06/2027", "LOT: A2506B"])
    assert f["mfg_date"].value == "2026-06-01" and f["lot_number"].value == "A2506B"
    assert d.status == DocumentStatus.AUTO_APPROVED, d.reasons


def test_buddhist_two_digit_year():
    assert normalize_year("68", TODAY) == (2025, True)
    assert normalize_year("27", TODAY) == (2027, False)


def test_ocr_digit_fix_lowers_confidence():
    fixed, n = fix_digits("1O/OI/2027")
    assert fixed == "10/01/2027" and n == 3
    f, _, d = run(["EXP 1O/OI/2027"])
    assert f["exp_date"].value == "2027-01-10"
    assert f["exp_date"].confidence < 0.90
    assert d.status == DocumentStatus.PENDING_REVIEW


def test_month_out_of_range_goes_to_review():
    f, rules, d = run(["EXP 2027-13-45"])
    assert any(r.rule_id == "R03" and not r.passed for r in rules)
    assert d.status == DocumentStatus.PENDING_REVIEW


def test_invalid_calendar_date():
    _, rules, d = run(["EXP 30/02/2027"])
    assert any(r.rule_id == "R04" and not r.passed for r in rules)
    assert not d.auto_approved


def test_exp_before_mfg():
    _, rules, d = run(["MFG 01/06/2027", "EXP 01/06/2026"])
    assert any(r.rule_id == "R05" and not r.passed for r in rules)
    assert not d.auto_approved


def test_mfg_in_future_and_shelf_life():
    _, rules, _ = run(["MFG 01/06/2027", "EXP 01/06/2045"])
    failed = {r.rule_id for r in rules if not r.passed}
    assert {"R06", "R07"} <= failed


def test_missing_exp():
    _, rules, d = run(["NET WT 500 g"])
    assert any(r.rule_id == "R01" and not r.passed for r in rules)
    assert not d.auto_approved


def test_label_without_value_is_flagged():
    f, _, d = run(["EXP: ##/##/####", "MFG 01/06/2026"])
    assert f["exp_date"].value is None and f["exp_date"].confidence == 0
    assert not d.auto_approved


def test_low_char_confidence_routes_to_review():
    line = TextLine("EXP 31/12/2027", 0.95, [0.99] * 10 + [0.55] + [0.99] * 3)
    _, _, d = run([line])
    assert d.status == DocumentStatus.PENDING_REVIEW
    assert any(r.startswith("LOW_CONFIDENCE:exp_date") for r in d.reasons)


def test_expired_is_warning_only():
    _, rules, d = run(["MFG 01/01/2025", "EXP 01/01/2026"])
    r09 = next(r for r in rules if r.rule_id == "R09")
    assert not r09.passed and r09.severity == "warning"
    assert d.auto_approved


def test_barcode_is_not_bb_label():
    f, _, _ = run(["BARCODE 8850000123", "EXP 31/12/2027"])
    assert f["exp_date"].value == "2027-12-31"


def test_unlabeled_dates_inferred():
    f, _, d = run(["12/03/2026", "12/03/2027"])
    assert f["exp_date"].value == "2027-03-12" and f["mfg_date"].value == "2026-03-12"
    assert f["exp_date"].inferred and not d.auto_approved  # ไม่มี label -> ให้คนยืนยัน


def test_find_dates_no_overlap():
    ds = find_dates("MFG 01/06/2026 EXP 01/06/2027")
    assert [d.raw for d in ds] == ["01/06/2026", "01/06/2027"]


def test_product_word_is_not_prod_label():
    f, _, _ = run(["PRODUCT OF THAILAND", "MFG 20/06/26", "EXP. 06/06/27"])
    assert f["mfg_date"].value == "2026-06-20" and f["exp_date"].value == "2027-06-06"


def test_inconsistent_separator_not_parsed_as_date():
    f, _, _ = run(["EXP. 1 16/07/28"])
    assert f["exp_date"].value == "2028-07-16"


def test_ocr_misread_labels():
    f, _, _ = run(["MEG 11-FEB-2026", "PR0D: 17/11/2568"])
    assert f["mfg_date"].value == "2026-02-11"
    f, _, _ = run(["PR0D: 17/11/2568 BEST BEF0RE 08/09/2569"])
    assert f["mfg_date"].value == "2025-11-17" and f["exp_date"].value == "2026-09-08"


def test_lot_code_starting_with_b8_is_not_bb_label():
    f, _, _ = run(["LOT N NO. B8699", "BB 01/02/2028"])
    assert f["lot_number"].value == "B8699" and f["exp_date"].value == "2028-02-01"


def test_label_value_on_next_line_is_inferred():
    f, _, d = run(["EXPIRY DATE:", "BARCODE 2027-06-24"])
    assert f["exp_date"].value == "2027-06-24" and f["exp_date"].inferred
    assert not d.auto_approved


@pytest.mark.parametrize("word", ["ผลิตภัณฑ์ของไทย", "ผลิตโดย บริษัท เอ จำกัด", "ผู้ผลิต บริษัท เอ", "MANUFACTURED BY ACME", "MEGA SALE"])
def test_words_that_are_not_mfg_label(word):
    f, _, _ = run([word, "EXP 31/12/2027"])
    assert "mfg_date" not in f


@pytest.mark.parametrize("word", ["EXPORT QUALITY", "EXPRESS DELIVERY"])
def test_words_that_are_not_exp_label(word):
    f, _, _ = run([word, "MFG 01/01/2026"])
    assert "exp_date" not in f


def test_lotion_is_not_lot_label():
    f, _, _ = run(["BODY LOTION 100ML", "EXP 31/12/2027"])
    assert "lot_number" not in f


@pytest.mark.parametrize("text", ["LOT NUMBER: A2506B", "BATCH NO. A2506B"])
def test_lot_label_words_are_not_the_value(text):
    f, _, _ = run([text])
    assert f["lot_number"].value == "A2506B"


def test_numeric_lot_code_is_kept_and_not_an_unclaimed_date():
    f, rules, _ = run(["MFG 01/01/2026", "EXP 01/01/2028", "LOT 120626"])
    assert f["lot_number"].value == "120626"
    assert not any(r.rule_id == "R10" and not r.passed for r in rules)


@pytest.mark.parametrize("text,mfg,exp", [
    ("MFG/EXP: 01/01/26 01/01/28", "2026-01-01", "2028-01-01"),
    ("ผลิต/หมดอายุ 01/01/69 01/01/71", "2026-01-01", "2028-01-01"),
])
def test_combined_label_assigns_dates_in_order(text, mfg, exp):
    f, _, _ = run([text])
    assert f["mfg_date"].value == mfg and f["exp_date"].value == exp


def test_combined_label_with_one_date_goes_to_review():
    f, _, d = run(["MFG/EXP: 01/01/28"])
    assert f["mfg_date"].value is None and not d.auto_approved


@pytest.mark.parametrize("text,iso", [
    ("EXPIRATION DATE 31/12/2027", "2027-12-31"),
    ("BEST BY 31/12/2027", "2027-12-31"),
    ("USE BEFORE 31/12/2027", "2027-12-31"),
    ("EX: 31/12/2027", "2027-12-31"),
    ("ควรบริโภคภายใน 31/12/2570", "2027-12-31"),
    ("หมดอายุ 31 ธ.ค 70", "2027-12-31"),
    ("หมดอายุ 31 ธค 70", "2027-12-31"),
    ("EXP 31AUGUST2027", "2027-08-31"),
])
def test_more_exp_labels_and_formats(text, iso):
    f, _, _ = run([text])
    assert f["exp_date"].value == iso


@pytest.mark.parametrize("line,name,value", [
    ("LOTU34563H", "lot_number", "U34563H"),
    ("BATCHA2506B", "lot_number", "A2506B"),
    ("EXPDEC2027", "exp_date", "2027-12-31"),
    ("EXP160CT2026", "exp_date", "2026-10-16"),
    ("MFGJAN2026", "mfg_date", "2026-01-01"),
    ("MFD 11.112025", "mfg_date", "2025-11-11"),
    ("EXP 3112.2027", "exp_date", "2027-12-31"),
])
def test_ocr_that_drops_spaces_and_separators(line, name, value):
    f, _, _ = run([line])
    assert f[name].value == value


@pytest.mark.parametrize("line,lot", [("101 W070K", "W070K"), ("[01: W7978A", "W7978A")])
def test_misread_lot_label(line, lot):
    f, _, _ = run([line])
    assert f["lot_number"].value == lot


@pytest.mark.parametrize("line", ["101 500", "ROOM 101 BANGKOK"])
def test_number_101_is_not_a_lot_label(line):
    f, _, _ = run([line])
    assert "lot_number" not in f


def test_date_with_lost_separator_before_year():
    f, _, _ = run(["MFD 0710 2569"])
    assert f["mfg_date"].value == "2026-10-07"


def test_lotion_without_space_is_not_lot_label():
    f, _, _ = run(["LOTION100ML", "EXP 31/12/2027"])
    assert "lot_number" not in f


def test_lowercase_month_mfg_is_not_unclaimed():
    _, rules, _ = run(["MFG Dec 2026", "EXP 01/01/2028"])
    assert not any(r.rule_id == "R10" and not r.passed for r in rules)


def test_unclaimed_date_routes_to_review():
    _, rules, d = run(["MFG 01/01/2026", "EXP 01/01/2028", "12/12/2027"])
    assert any(r.rule_id == "R10" and not r.passed for r in rules)
    assert not d.auto_approved
