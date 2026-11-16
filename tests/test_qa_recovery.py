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

    original_checkpoint_bytes = (state / "checkpoints" / "baseline").glob("*.json")
    checkpoint_path = next(original_checkpoint_bytes)
    original_checkpoint = checkpoint_path.read_bytes()
    original_prediction = (
        Path(config["output_dir"]) / "magic" / "baseline" / "predictions.jsonl"
    )
    original_prediction_bytes = original_prediction.read_bytes()
    original_score_bytes = (
        Path(config["output_dir"]) / "magic" / "baseline" / "scores.jsonl"
    ).read_bytes()
    original_summary_bytes = (
        Path(config["output_dir"]) / "magic" / "baseline" / "summary.json"
    ).read_bytes()

    after = recover_qa_parse_errors(config)
    assert model.calls == calls_before  # No API calls on recovery.
    assert [r["status"] for r in after] == ["ok", "parse_error", "parse_error"]
    assert [r["accqa"] for r in after] == [1, 0, 0]

    # Checkpoints and original exported results are byte-for-byte unchanged.
    assert checkpoint_path.read_bytes() == original_checkpoint
    assert original_prediction.read_bytes() == original_prediction_bytes
    assert (
        Path(config["output_dir"]) / "magic" / "baseline" / "scores.jsonl"
    ).read_bytes() == original_score_bytes
    assert (
        Path(config["output_dir"]) / "magic" / "baseline" / "summary.json"
    ).read_bytes() == original_summary_bytes

    dest = Path(config["output_dir"]) / "recovered"
    predictions = [
        json.loads(line)
        for line in (dest / "magic" / "baseline" / "predictions.jsonl").read_text().splitlines()
    ]
    assert len(predictions) == 1
    assert predictions[0]["predicted_answer"] == "A"
    assert predictions[0]["recovered_from_saved_response"] is True
    assert predictions[0]["raw_response"] == "I examined the evidence.\\n\\nA"
    summary = json.loads((dest / "magic" / "baseline" / "summary.json").read_text())
    assert summary["correct"] == 1 and summary["count"] == 1
    report = json.loads((dest / "recovery_summary.json").read_text())
    assert report["recovered_count"] == 1
    assert report["original"]["by_condition"]["baseline"]["correct"] == 0
    assert report["recovered"]["by_condition"]["baseline"]["correct"] == 1

    first_export = (dest / "magic" / "baseline" / "predictions.jsonl").read_bytes()
    assert recover_qa_parse_errors(config) == after  # Idempotent.
    assert (dest / "magic" / "baseline" / "predictions.jsonl").read_bytes() == first_export
