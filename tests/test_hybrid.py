from math_structure.data_v2 import generate_examples
from scripts.evaluate_hybrid import score_routing_split


def test_oracle_and_gold_route_are_perfect() -> None:
    rows = generate_examples(24, "seen", __import__("random").Random(7))
    completions = [row["routing_target"] for row in rows]
    metric, diagnostics = score_routing_split(
        rows,
        completions,
        train_size=24,
        split="seen",
        direct_accuracies={
            "baseline_direct_accuracy": 0.25,
            "structure_direct_accuracy": 0.5,
        },
    )

    assert metric["oracle_plugin_accuracy"] == 1.0
    assert metric["predicted_plugin_accuracy"] == 1.0
    assert metric["structure_accuracy"] == 1.0
    assert metric["tool_accuracy"] == 1.0
    assert metric["argument_accuracy"] == 1.0
    assert metric["plugin_execution_success_rate"] == 1.0
    assert metric["final_answer_given_correct_tool"] == 1.0
    assert metric["final_answer_given_correct_tool_and_arguments"] == 1.0
    assert metric["plugin_success_given_correct_tool"] == 1.0
    assert all(not row["error_category"] for row in diagnostics)


def test_invalid_predicted_arguments_are_diagnosed() -> None:
    row = generate_examples(1, "seen", __import__("random").Random(2))[0]
    completion = (
        f"STRUCTURE: {row['structure']}\n"
        f"TOOL: {row['tool']}\n"
        "ARGUMENTS: {}"
    )
    metric, diagnostics = score_routing_split(
        [row],
        [completion],
        train_size=1,
        split="seen",
        direct_accuracies={
            "baseline_direct_accuracy": 0.0,
            "structure_direct_accuracy": 0.0,
        },
    )

    assert metric["oracle_plugin_accuracy"] == 1.0
    assert metric["predicted_plugin_accuracy"] == 0.0
    assert diagnostics[0]["error_category"] == "invalid_arguments"
