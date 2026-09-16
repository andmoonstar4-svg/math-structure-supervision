import json
from pathlib import Path

from scripts.evaluate import (
    analyze_label_mapping,
    build_comparison_rows,
    build_confusion_rows,
    conditional_execution_metrics,
    discover_runs,
    score_split,
)
from scripts.train_common import (
    make_prompt,
    make_target,
    parse_prediction,
    parse_routing_prediction,
)


EXAMPLE = {
    "id": "example_1",
    "problem": "If x + y = 5 and xy = 6, find x^2 + y^2.",
    "answer": "13",
    "structure": "symmetric_polynomial_identity",
    "tool": "expand_symmetric_identity",
    "baseline_target": "13",
    "oracle_target": "13",
    "tool_args": {"mode": "symmetric_square_sum", "sum": 5, "product": 6},
    "structured_target": (
        "STRUCTURE: symmetric_polynomial_identity\n"
        "TOOL: expand_symmetric_identity\n"
        "ANSWER: 13"
    ),
    "routing_target": (
        "STRUCTURE: symmetric_polynomial_identity\n"
        "TOOL: expand_symmetric_identity\n"
        'ARGUMENTS: {"mode":"symmetric_square_sum","product":6,"sum":5}'
    ),
}


def test_prompts_targets_and_prediction_parser() -> None:
    assert make_target(EXAMPLE, "baseline") == "13"
    assert make_target(EXAMPLE, "structure_aware") == EXAMPLE["structured_target"]
    assert make_target(EXAMPLE, "oracle") == "13"
    assert "Return only the final answer" in make_prompt(EXAMPLE, "baseline")
    assert "exactly three lines" in make_prompt(EXAMPLE, "structure_aware")
    oracle_prompt = make_prompt(EXAMPLE, "oracle")
    assert "Structure: symmetric_polynomial_identity" in oracle_prompt
    assert "Tool: expand_symmetric_identity" in oracle_prompt
    assert oracle_prompt.endswith("Answer:")

    baseline = parse_prediction("13\nExtra text", "baseline")
    assert baseline == {"answer": "13", "structure": None, "tool": None}

    structured = parse_prediction(EXAMPLE["structured_target"], "structure_aware")
    assert structured == {
        "answer": "13",
        "structure": "symmetric_polynomial_identity",
        "tool": "expand_symmetric_identity",
    }
    assert parse_prediction("13\nExtra text", "oracle") == baseline


def test_routing_prompt_target_and_prediction_parser() -> None:
    prompt = make_prompt(EXAMPLE, "tool_routing")
    assert "do not return an answer" in prompt
    assert "ARGUMENTS: <json object>" in prompt
    assert make_target(EXAMPLE, "tool_routing") == EXAMPLE["routing_target"]

    parsed = parse_routing_prediction(
        EXAMPLE["routing_target"] + "\nThis trailing text is ignored."
    )
    assert parsed["structure"] == EXAMPLE["structure"]
    assert parsed["tool"] == EXAMPLE["tool"]
    assert parsed["tool_args"] == EXAMPLE["tool_args"]

    invalid = parse_routing_prediction(
        "STRUCTURE: transformed_expression\n"
        "TOOL: expansion\n"
        "ARGUMENTS: not-json"
    )
    assert invalid["structure"] == "transformed_expression"
    assert invalid["tool"] == "expansion"
    assert invalid["tool_args"] is None
    assert invalid["tool_args_text"] == "not-json"


def test_scoring_reports_structure_metrics_only_when_supervised() -> None:
    structured_metric, _ = score_split(
        [EXAMPLE], [EXAMPLE["structured_target"]], "structure_aware", 50, "unseen"
    )
    assert structured_metric["exact_answer_accuracy"] == 1.0
    assert structured_metric["structure_accuracy"] == 1.0
    assert structured_metric["tool_accuracy"] == 1.0

    baseline_metric, _ = score_split([EXAMPLE], ["13"], "baseline", 50, "seen")
    assert baseline_metric["exact_answer_accuracy"] == 1.0
    assert baseline_metric["structure_accuracy"] is None
    assert baseline_metric["tool_accuracy"] is None

    oracle_metric, oracle_predictions = score_split(
        [EXAMPLE], ["13"], "oracle", 50, "seen"
    )
    assert oracle_metric["exact_answer_accuracy"] == 1.0
    assert oracle_metric["structure_accuracy"] is None
    assert oracle_predictions[0]["gold_structure"] == EXAMPLE["structure"]
    assert oracle_predictions[0]["gold_tool"] == EXAMPLE["tool"]


def test_conditional_execution_metrics() -> None:
    predictions = [
        {"answer_correct": True, "structure_correct": True, "tool_correct": True},
        {"answer_correct": False, "structure_correct": True, "tool_correct": False},
        {"answer_correct": True, "structure_correct": False, "tool_correct": True},
        {"answer_correct": False, "structure_correct": False, "tool_correct": False},
    ]
    metrics = conditional_execution_metrics(predictions)

    assert metrics["answer_given_correct_structure"] == 0.5
    assert metrics["answer_given_correct_tool"] == 1.0
    assert metrics["answer_given_correct_structure_and_tool"] == 1.0
    assert metrics["answer_given_structure_or_tool_incorrect"] == 1 / 3
    assert metrics["structure_or_tool_incorrect_n"] == 3


def test_label_mapping_analysis_detects_redundancy() -> None:
    mapping = analyze_label_mapping(
        [
            {"structure": "factorization", "tool": "factor"},
            {"structure": "substitution", "tool": "substitute"},
            {"structure": "factorization", "tool": "factor"},
        ]
    )
    assert mapping["structure_to_tools"] == {
        "factorization": ["factor"],
        "substitution": ["substitute"],
    }
    assert mapping["tool_to_structures"]["factor"] == ["factorization"]
    assert mapping["is_many_to_many"] is False
    assert mapping["is_one_to_one"] is True
    assert "redundant" in mapping["warning"]

    not_one_to_one = analyze_label_mapping(
        [
            {"structure": "factorization", "tool": "factor"},
            {"structure": "factorization", "tool": "expand"},
        ]
    )
    assert not_one_to_one["is_one_to_one"] is False
    assert not_one_to_one["warning"] is None


def test_wide_comparison_table() -> None:
    metrics = [
        {
            "setting": "baseline",
            "train_size": 50,
            "template_split": "seen",
            "exact_answer_accuracy": 0.1,
        },
        {
            "setting": "structure_aware",
            "train_size": 50,
            "template_split": "seen",
            "exact_answer_accuracy": 0.2,
            "structure_accuracy": 0.8,
            "tool_accuracy": 0.7,
        },
        {
            "setting": "oracle",
            "train_size": 50,
            "template_split": "seen",
            "exact_answer_accuracy": 0.3,
        },
    ]
    conditional = [
        {
            "train_size": 50,
            "split": "seen",
            "answer_given_correct_structure": 0.25,
            "answer_given_correct_tool": 0.2,
            "answer_given_correct_structure_and_tool": 0.3,
            "answer_given_structure_or_tool_incorrect": 0.1,
        }
    ]
    row = build_comparison_rows(metrics, conditional)[0]
    assert row["baseline_answer_accuracy"] == 0.1
    assert row["structure_answer_accuracy"] == 0.2
    assert row["oracle_answer_accuracy"] == 0.3
    assert row["answer_given_correct_structure_and_tool"] == 0.3


def test_confusion_rows_keep_missing_predictions_visible() -> None:
    predictions = [
        {
            "setting": "structure_aware",
            "train_size": 50,
            "template_split": "unseen",
            "gold_structure": "two_linear_relations",
            "predicted_structure": "two_linear_relations",
        },
        {
            "setting": "structure_aware",
            "train_size": 50,
            "template_split": "unseen",
            "gold_structure": "two_linear_relations",
            "predicted_structure": "",
        },
        {
            "setting": "baseline",
            "train_size": 50,
            "template_split": "unseen",
            "gold_structure": "ignored",
            "predicted_structure": "ignored",
        },
    ]
    rows = build_confusion_rows(
        predictions, "gold_structure", "predicted_structure"
    )
    assert len(rows) == 2
    assert sum(row["count"] for row in rows) == 2
    assert {row["predicted_structure"] for row in rows} == {
        "two_linear_relations",
        "<missing>",
    }


def test_run_discovery_ignores_incomplete_directories(tmp_path: Path) -> None:
    complete = tmp_path / "baseline" / "train_50"
    incomplete = tmp_path / "structure_aware" / "train_100"
    complete.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    (complete / "adapter_config.json").write_text(json.dumps({}), encoding="utf-8")

    assert discover_runs(tmp_path, ["baseline", "structure_aware"], None) == [
        ("baseline", 50, complete)
    ]
