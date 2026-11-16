"""Regression coverage for bounded final-line QA parsing and offline recovery."""
import json
from pathlib import Path

import pytest

from glint.evaluation import evaluate
from glint.evaluation.parsing import parse_answer_response
from glint.evaluation.progressive import recover_qa_parse_errors
from glint.evaluation.results import load_checkpoint, save_checkpoint
from glint.evaluation.types import ParseError
from glint.models import ModelResponse
from test_evaluation import config


def test_final_line_answer_after_explanation():
    text = (
        "Based on the video, the first black-suited card is a spade.\n"
        "The card immediately after it is a heart.\n\nB"
    )
    assert parse_answer_response(text) == "B"
    assert parse_answer_response("Reasoning here.\n\nAnswer: **c**.") == "C"
    assert parse_answer_response("Reasoning here.\nD.") == "D"
    assert parse_answer_response('{"answer":"A"}') == "A"
    assert parse_answer_response("**B**") == "B"


@pytest.mark.parametrize("ambiguous", [
    "Reasoning suggests A, but B could also fit.",
    "B or C",
    "Reasoning here.\nA or B",
    "Reasoning here.\nFinal answer: B or C",
    '{"answer":"A"} and some text\nB',
    "Reasoning here.\nAB",
    "",
])
def test_ambiguous_qa_remains_rejected(ambiguous):
    with pytest.raises(ParseError):
        parse_answer_response(ambiguous)


def test_recover_saved_parse_error_without_model_calls(config):
    config["task"] = "progressive_qa"

    class Malformed:
        calls = 0
        def generate(self, messages, **kwargs):
            self.calls += 1
            return ModelResponse("no clear answer", "fake")

    model = Malformed()
    before = evaluate(config, model=model)
    assert len(before) == 3 and all(r["status"] == "parse_error" for r in before)
    calls_before = model.calls

    state = Path(config["output_dir"]) / "runtime"
    original = load_checkpoint(state, "magic_0001_q01", "baseline")
    original["raw_response"] = "I examined the evidence.\n\nA"
    save_checkpoint(state, original)

    after = recover_qa_parse_errors(config)
    assert model.calls == calls_before  # No API calls on recovery.
    assert [r["status"] for r in after] == ["ok", "parse_error", "parse_error"]
    assert [r["accqa"] for r in after] == [1, 0, 0]

    restored = load_checkpoint(state, "magic_0001_q01", "baseline")
    assert restored["predicted_answer"] == "A"
    assert restored["recovered_from_saved_response"] is True
    assert restored["raw_response"] == "I examined the evidence.\n\nA"
    summary = json.loads(
        (Path(config["output_dir"]) / "magic" / "baseline" / "summary.json").read_text()
    )
    assert summary["correct"] == 1 and summary["count"] == 1
    assert recover_qa_parse_errors(config) == after  # Idempotent.
