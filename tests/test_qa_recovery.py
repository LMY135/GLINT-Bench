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



def _make_original_export(root, category, qid, condition, status, raw, predicted=None, accqa=0.0):
    directory = root / category / condition
    directory.mkdir(parents=True, exist_ok=True)
    prediction = {
        "question_id": qid,
        "category": category,
        "condition": condition,
        "status": status,
        "raw_response": raw,
        "predicted_answer": predicted,
        "correct": bool(accqa),
        "history": [{"text": raw}],
    }
    score = {
        "question_id": qid, "category": category, "condition": condition,
        "status": status, "accqa": accqa,
    }
    (directory / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
    (directory / "scores.jsonl").write_text(json.dumps(score) + "\n")


def test_offline_recovery_without_old_prompt_or_metadata(tmp_path):
    output = tmp_path / "old_v1_run"
    qa_path = tmp_path / "new_qa.jsonl"
    qa_path.write_text(
        json.dumps({"question_id": "magic_0008_q02", "category": "magic", "answer": "B"})
        + "\n" + json.dumps({"question_id": "unrelated_new_q", "category": "game", "answer": "D"}) + "\n"
    )
    for condition in ("baseline", "temporal_gt", "spatiotemporal_gt"):
        if condition == "baseline":
            _make_original_export(output, "magic", "magic_0008_q02", condition,
                                  "parse_error", "The suit is Hearts.\n\nB")
        else:
            _make_original_export(output, "magic", "magic_0008_q02", condition,
                                  "call_error", None)

    # No run_config, checkpoints, video files or metadata are present.
    originals = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    config = {"output_dir": str(output), "qa_path": str(qa_path),
              "categories": ["magic"], "model": "different-model"}
    rows = recover_qa_parse_errors(config)
    assert [r["accqa"] for r in rows] == [1, 0, 0]
    assert all(p.read_bytes() == content for p, content in originals.items())

    destination = output / "recovered"
    restored = json.loads(
        (destination / "magic" / "baseline" / "predictions.jsonl").read_text().strip()
    )
    assert restored["predicted_answer"] == "B"
    assert restored["recovered_from_saved_response"] is True
    assert restored["raw_response"] == "The suit is Hearts.\n\nB"
    summary = json.loads((destination / "recovery_summary.json").read_text())
    assert summary["recovered_count"] == 1
    assert summary["original"]["by_condition"]["baseline"]["accqa"] == 0
    assert summary["recovered"]["by_condition"]["baseline"]["accqa"] == 1
    assert (destination / "runtime" / "result_index.json").is_file()

    first = (destination / "magic" / "baseline" / "predictions.jsonl").read_bytes()
    recover_qa_parse_errors(config)
    assert (destination / "magic" / "baseline" / "predictions.jsonl").read_bytes() == first
    assert all(p.read_bytes() == content for p, content in originals.items())


def test_recovery_all_categories_and_preserve_existing_scores(tmp_path):
    output = tmp_path / "multi_category_run"
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text("\n".join([
        json.dumps({"question_id": "m1", "category": "magic", "answer": "A"}),
        json.dumps({"question_id": "g1", "category": "game", "answer": "C"}),
    ]) + "\n")
    for category, qid, answer in (("magic", "m1", "A"), ("game", "g1", "C")):
        for condition in ("baseline", "temporal_gt", "spatiotemporal_gt"):
            if condition == "baseline":
                _make_original_export(output, category, qid, condition,
                                      "parse_error", "My explanation\n" + answer)
            else:
                _make_original_export(output, category, qid, condition,
                                      "ok", answer, answer, 1.0)

    config = {"output_dir": str(output), "qa_path": str(qa_path), "categories": []}
    source_bytes = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    rows = recover_qa_parse_errors(config)
    assert len(rows) == 6
    assert all(row["accqa"] == 1 for row in rows)
    assert all(p.read_bytes() == content for p, content in source_bytes.items())
    report = json.loads((output / "recovered" / "recovery_summary.json").read_text())
    assert report["recovered_count"] == 2
    assert set(report["recovered"]["by_category"]) == {"magic", "game"}


def test_missing_ground_truth_refuses_to_change_outputs(tmp_path):
    output = tmp_path / "old_run"
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text(json.dumps({"question_id": "different", "category": "magic", "answer": "A"}) + "\n")
    for condition in ("baseline", "temporal_gt", "spatiotemporal_gt"):
        _make_original_export(output, "magic", "missing", condition,
                              "parse_error", "Reasoning\nB")
    with pytest.raises(ValueError, match="Cannot safely score"):
        recover_qa_parse_errors({"output_dir": str(output), "qa_path": str(qa_path),
                                 "categories": ["magic"]})
    assert not (output / "recovered").exists()


def test_historical_call_error_retry_preserves_original_and_resumes(config):
    from glint.evaluation.progressive import retry_historical_call_errors
    from glint.evaluation.results import load_checkpoint
    from glint.evaluation import evaluate

    config["task"] = "progressive_qa"
    config["categories"] = ["magic"]

    class Failing:
        calls = 0
        def generate(self, messages, **kwargs):
            self.calls += 1
            raise RuntimeError("402 prepayment credits depleted")

    old_model = Failing()
    evaluate(config, model=old_model)
    assert old_model.calls == 3
    output = Path(config["output_dir"])
    snapshots = {str(p.relative_to(output)): p.read_bytes()
                 for p in output.rglob("*") if p.is_file()}

    # Changes to unrelated QA files/metadata must not stop a historical retry.
    with open(config["qa_path"], "a", encoding="utf-8") as stream:
        stream.write("\n")
    with open(config["metadata_path"], "a", encoding="utf-8") as stream:
        stream.write("\n")

    class Success:
        calls = 0
        def generate(self, messages, **kwargs):
            self.calls += 1
            return ModelResponse("A", "fake")

    new_model = Success()
    retried = retry_historical_call_errors(config, model=new_model)
    assert new_model.calls == 3
    assert len(retried) == 3 and all(x["accqa"] == 1 for x in retried)
    dest = output / "retried_call_errors"
    assert (dest / "magic" / "baseline" / "summary.json").is_file()
    assert load_checkpoint(dest / "runtime", "magic_0001_q01", "baseline")["status"] == "ok"
    assert all((output / key).read_bytes() == contents for key, contents in snapshots.items())

    # Restart does not make paid API calls again.
    retried_again = retry_historical_call_errors(config, model=new_model)
    assert retried_again == retried
    assert new_model.calls == 3


def test_historical_retry_only_failed_conditions(config):
    from glint.evaluation.progressive import retry_historical_call_errors

    config["task"] = "progressive_qa"
    config["categories"] = ["magic"]
    output = Path(config["output_dir"])

    for condition in ("baseline", "temporal_gt", "spatiotemporal_gt"):
        status = "ok" if condition == "baseline" else "call_error"
        _make_original_export(output, "magic", "magic_0001_q01", condition,
                              status, "A" if status == "ok" else None,
                              "A" if status == "ok" else None,
                              1.0 if status == "ok" else 0.0)
    class Success:
        calls = 0
        def generate(self, messages, **kwargs):
            self.calls += 1
            return ModelResponse("A", "fake")

    model = Success()
    scores = retry_historical_call_errors(config, model=model)
    assert model.calls == 2
    assert all(row["accqa"] == 1 for row in scores)
    copied = json.loads((output / "retried_call_errors" / "magic" /
                         "baseline" / "predictions.jsonl").read_text().strip())
    assert copied["status"] == "ok" and copied["predicted_answer"] == "A"
