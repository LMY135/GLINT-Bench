"""Regression tests for conservative offline QA final-answer recovery."""
import pytest

from glint.evaluation.parsing import parse_answer_response, parse_answer_with_method
from glint.evaluation.types import ParseError


@pytest.mark.parametrize("text,answer,method", [
    ("A", "A", "plain_option"),
    ("Answer: **d**", "D", "plain_option"),
    ("<answer>C", "C", "answer_tag"),
    ("</think><answer>D", "D", "answer_tag"),
    ("</think><answer><think><answer>A", "A", "answer_tag"),
    ("<answer>B</answer>", "B", "answer_tag"),
    ('{"answer":"C","evidence_windows":[]}.', "C", "json_answer"),
    ('{"answer":"D"} (Note: checked carefully.)', "D", "json_answer"),
    ('[{"answer":"A"}]', "A", "json_answer"),
    ("Some explanation\nB", "B", "final_line"),
])
def test_explicit_final_answer_variants(text, answer, method):
    assert parse_answer_with_method(text) == (answer, method)
    assert parse_answer_response(text) == answer


@pytest.mark.parametrize("text", [
    "",
    "A or B",
    "I think B but C is the answer",
    "<think>A might be right, but B might be too.",
    "<think>A</think><answer>B</answer><answer>C</answer>",
    "</think><answer><think>unfinished",
    "<answer>Still reasoning about C",
    '{"answer":"A"} and some prose',
    '{"answer":"A"}{"answer":"B"}',
    '{"answer":"A","answer2":"B"} and some prose',
    '[{"answer":"A"},{"answer":"B"}]',
    'AB',
])
def test_ambiguous_and_unfinished_rejected(text):
    with pytest.raises(ParseError):
        parse_answer_response(text)
