"""Tests for refusal classification + aggregation (mass/refusals.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mass"))
from refusals import (classify, summarize, STATUS_ANSWERED, STATUS_REFUSED,  # noqa: E402
                      STATUS_EMPTY, STATUS_ERROR, STATUS_UNKNOWN)


def test_answered_variants():
    assert classify("B", 4) == (1, STATUS_ANSWERED, "")
    assert classify("  A) because it is fair", 4)[1] == STATUS_ANSWERED
    assert classify("<think></think>C", 4) == (2, STATUS_ANSWERED, "")  # reasoning model, empty think
    assert classify("<think>weigh options</think>D", 5)[0] == 3


def test_out_of_range_letter_is_not_answered():
    # 'E' when only 4 options -> not a valid answer
    assert classify("E", 4)[0] is None


def test_empty_and_error_are_transient():
    assert classify("", 4) == (None, STATUS_EMPTY, "")
    assert classify("   ", 4)[1] == STATUS_EMPTY
    assert classify("<think></think>", 4)[1] == STATUS_EMPTY
    assert classify("__ERROR__ HTTP 500", 4)[1] == STATUS_ERROR


def test_genuine_refusal_captures_reason():
    idx, st, reason = classify(
        "I won't pick a letter here — I don't hold personal political views.", 4)
    assert idx is None and st == STATUS_REFUSED
    assert "personal political views" in reason
    assert classify("As an AI, I can't take a partisan stance on this.", 4)[1] == STATUS_REFUSED
    assert classify("I'm not able to answer that.", 4)[1] == STATUS_REFUSED


def test_unknown_when_no_letter_no_cue():
    # non-empty, no valid letter, no refusal cue -> unknown (inspect / rerun)
    assert classify("42 purple", 4)[1] == STATUS_UNKNOWN


def test_summarize_flags_impacted_dimension():
    # 20 items load on 'economic'; 4 refused (20%) -> impacted. 'social' 1 refusal -> not.
    rows, item_dims = [], {}
    for i in range(20):
        iid = f"E{i}"
        item_dims[iid] = {"economic"}
        status = STATUS_REFUSED if i < 4 else STATUS_ANSWERED
        rows.append({"item_id": iid, "n_options": 4, "status": status,
                     "raw": "I won't state a personal view" if i < 4 else "A",
                     "chosen_idx": None if i < 4 else 0})
    rows.append({"item_id": "S0", "n_options": 4, "status": STATUS_REFUSED,
                 "raw": "no personal opinion", "chosen_idx": None})
    item_dims["S0"] = {"social"}
    rep = summarize(rows, item_dims, ["economic", "social", "religion"])
    assert rep["genuine_refusals"] == 5
    assert rep["per_dimension"]["economic"]["leaning_may_be_impacted"] is True
    assert rep["per_dimension"]["social"]["leaning_may_be_impacted"] is False
    assert rep["per_dimension"]["religion"]["scored_or_refused"] == 0


def test_summarize_derives_status_for_legacy_rows():
    # rows without a 'status' field: use chosen_idx / classify(raw)
    rows = [{"item_id": "X", "n_options": 4, "chosen_idx": 2, "raw": "C"},
            {"item_id": "Y", "n_options": 4, "chosen_idx": None, "raw": ""}]
    item_dims = {"X": {"economic"}, "Y": {"economic"}}
    rep = summarize(rows, item_dims, ["economic"])
    assert rep["status_counts"].get(STATUS_ANSWERED) == 1
    assert rep["status_counts"].get(STATUS_EMPTY) == 1
