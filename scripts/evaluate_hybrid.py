"""Evaluate direct generation and deterministic-plugin routing side by side."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

try:
    from .evaluate import (
        _base_model_for,
        _model_dtype_for,
        discover_runs,
        generate_predictions,
        write_csv,
    )
    from .train_common import (
        DEFAULT_MODEL,
        DEFAULT_TRAIN_SIZES,
        load_jsonl,
        normalize_answer,
        normalize_label,
        parse_routing_prediction,
    )
except ImportError:  # Allows `python scripts/evaluate_hybrid.py`.
    from evaluate import (
        _base_model_for,
        _model_dtype_for,
        discover_runs,
        generate_predictions,
        write_csv,
    )
    from train_common import (
        DEFAULT_MODEL,
        DEFAULT_TRAIN_SIZES,
        load_jsonl,
        normalize_answer,
        normalize_label,
        parse_routing_prediction,
    )

from math_structure.tools import (  # noqa: E402
    PluginError,
    ToolArgumentError,
    ToolExecutionError,
    execute_tool,
    tool_args_semantically_equivalent,
    validate_tool_args,
)


METRIC_FIELDS = [
    "train_size",
    "split",
    "n",
    "baseline_direct_accuracy",
    "structure_direct_accuracy",
    "predicted_plugin_accuracy",
    "oracle_plugin_accuracy",
    "structure_accuracy",
    "tool_accuracy",
    "argument_accuracy",
    "argument_exact_match_accuracy",
    "argument_semantic_validity_rate",
    "argument_parse_rate",
    "plugin_execution_success_rate",
    "final_answer_given_correct_tool",
    "final_answer_given_correct_tool_and_arguments",
    "plugin_success_given_correct_tool",
]

PREDICTION_FIELDS = [
    "id",
    "train_size",
    "split",
    "problem",
    "gold_structure",
    "predicted_structure",
    "structure_correct",
    "gold_tool",
    "predicted_tool",
    "tool_correct",
    "gold_tool_args",
    "predicted_tool_args",
    "argument_parse_success",
    "argument_schema_valid",
    "argument_exact_match",
    "argument_semantically_equivalent",
    "plugin_execution_success",
    "gold_answer",
    "plugin_answer",
    "final_answer_correct",
    "oracle_plugin_answer",
    "oracle_plugin_correct",
    "error_category",
    "raw_prediction",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data_v2_tools"))
    parser.add_argument(
        "--routing-output-dir", type=Path, default=Path("outputs/qwen_v2_hybrid")
    )
    parser.add_argument(
        "--direct-predictions-csv",
        type=Path,
        default=Path("results_qwen_v2/predictions.csv"),
    )
    parser.add_argument(
        "--metrics-csv", type=Path, default=Path("results_hybrid/metrics.csv")
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("results_hybrid/predictions.csv"),
    )
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAIN_SIZES),
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--model-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help=f"Base model dtype (metadata fallback model: {DEFAULT_MODEL}).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    return parser


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _compact_json(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_direct_accuracies(
    path: Path,
    test_sets: dict[str, list[dict[str, Any]]],
    train_sizes: list[int],
) -> dict[tuple[int, str], dict[str, float]]:
    """Reuse and validate third-round per-example direct predictions."""

    expected_rows = {
        (split, str(row["id"])): row
        for split, rows in test_sets.items()
        for row in rows
    }
    totals: dict[tuple[str, int, str], list[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for prediction in csv.DictReader(handle):
            setting = prediction.get("setting", "")
            if setting not in {"baseline", "structure_aware"}:
                continue
            train_size = int(prediction["train_size"])
            if train_size not in train_sizes:
                continue
            split = prediction["template_split"]
            key = (split, prediction["id"])
            if key not in expected_rows:
                raise ValueError(f"Direct prediction has unknown example {key}")
            gold = expected_rows[key]
            if prediction["problem"] != str(gold["problem"]):
                raise ValueError(f"Problem mismatch for direct prediction {key}")
            if normalize_answer(prediction["gold_answer"]) != normalize_answer(
                gold["answer"]
            ):
                raise ValueError(f"Answer mismatch for direct prediction {key}")
            counts = totals.setdefault((setting, train_size, split), [0, 0])
            counts[0] += int(_as_bool(prediction["answer_correct"]))
            counts[1] += 1

    accuracies: dict[tuple[int, str], dict[str, float]] = {}
    for train_size in train_sizes:
        for split, rows in test_sets.items():
            item: dict[str, float] = {}
            for setting, name in (
                ("baseline", "baseline_direct_accuracy"),
                ("structure_aware", "structure_direct_accuracy"),
            ):
                correct, count = totals.get((setting, train_size, split), [0, 0])
                if count != len(rows):
                    raise ValueError(
                        f"Expected {len(rows)} {setting} predictions for "
                        f"n={train_size}, {split}; found {count}"
                    )
                item[name] = correct / count
            accuracies[(train_size, split)] = item
    return accuracies


def _conditional_rate(rows: list[dict[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    return sum(bool(row[field]) for row in rows) / len(rows)


def score_routing_split(
    rows: list[dict[str, Any]],
    completions: list[str],
    train_size: int,
    split: str,
    direct_accuracies: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute predicted routes and separate recognition, routing, and execution."""

    if len(rows) != len(completions):
        raise ValueError("Number of routing predictions does not match test rows")

    diagnostics: list[dict[str, Any]] = []
    for index, (row, completion) in enumerate(zip(rows, completions)):
        parsed = parse_routing_prediction(completion)
        predicted_structure = parsed["structure"] or ""
        predicted_tool = parsed["tool"] or ""
        predicted_args = parsed["tool_args"]
        structure_correct = normalize_label(predicted_structure) == normalize_label(
            row["structure"]
        )
        tool_correct = normalize_label(predicted_tool) == normalize_label(row["tool"])
        argument_parse_success = isinstance(predicted_args, dict)
        argument_exact = argument_parse_success and _compact_json(
            predicted_args
        ) == _compact_json(row["tool_args"])
        argument_semantic = argument_parse_success and tool_args_semantically_equivalent(
            row["tool"], predicted_args, row["tool_args"]
        )

        argument_schema_valid = False
        plugin_success = False
        plugin_answer = ""
        execution_failure = False
        if predicted_tool and argument_parse_success:
            try:
                validate_tool_args(predicted_tool, predicted_args)
                argument_schema_valid = True
            except ToolArgumentError:
                argument_schema_valid = False
            if argument_schema_valid:
                try:
                    plugin_answer = execute_tool(predicted_tool, predicted_args)
                    plugin_success = True
                except ToolExecutionError:
                    execution_failure = True
                except ToolArgumentError:
                    argument_schema_valid = False

        final_correct = plugin_success and normalize_answer(
            plugin_answer
        ) == normalize_answer(row["answer"])
        oracle_answer = execute_tool(row["tool"], row["tool_args"])
        oracle_correct = normalize_answer(oracle_answer) == normalize_answer(row["answer"])
        if not oracle_correct:
            raise ValueError(f"Oracle plugin failed for {row.get('id', index)}")

        error_category = ""
        if not final_correct:
            if not structure_correct:
                error_category = "wrong_structure"
            elif not tool_correct:
                error_category = "wrong_tool"
            elif not argument_parse_success or not argument_schema_valid:
                error_category = "invalid_arguments"
            elif not argument_semantic:
                error_category = "valid_but_wrong_arguments"
            elif execution_failure or not plugin_success:
                error_category = "plugin_execution_failure"
            else:
                error_category = "plugin_answer_mismatch"

        diagnostics.append(
            {
                "id": row.get("id", index),
                "train_size": train_size,
                "split": split,
                "problem": row["problem"],
                "gold_structure": row["structure"],
                "predicted_structure": predicted_structure,
                "structure_correct": structure_correct,
                "gold_tool": row["tool"],
                "predicted_tool": predicted_tool,
                "tool_correct": tool_correct,
                "gold_tool_args": _compact_json(row["tool_args"]),
                "predicted_tool_args": _compact_json(predicted_args)
                or parsed["tool_args_text"],
                "argument_parse_success": argument_parse_success,
                "argument_schema_valid": argument_schema_valid,
                "argument_exact_match": argument_exact,
                "argument_semantically_equivalent": argument_semantic,
                "plugin_execution_success": plugin_success,
                "gold_answer": row["answer"],
                "plugin_answer": plugin_answer,
                "final_answer_correct": final_correct,
                "oracle_plugin_answer": oracle_answer,
                "oracle_plugin_correct": oracle_correct,
                "error_category": error_category,
                "raw_prediction": completion,
            }
        )

    count = len(diagnostics)
    tool_correct_rows = [row for row in diagnostics if row["tool_correct"]]
    tool_and_args_rows = [
        row
        for row in diagnostics
        if row["tool_correct"] and row["argument_semantically_equivalent"]
    ]
    metric = {
        "train_size": train_size,
        "split": split,
        "n": count,
        **direct_accuracies,
        "predicted_plugin_accuracy": sum(
            row["final_answer_correct"] for row in diagnostics
        )
        / count,
        "oracle_plugin_accuracy": sum(
            row["oracle_plugin_correct"] for row in diagnostics
        )
        / count,
        "structure_accuracy": sum(row["structure_correct"] for row in diagnostics)
        / count,
        "tool_accuracy": sum(row["tool_correct"] for row in diagnostics) / count,
        "argument_accuracy": sum(
            row["argument_semantically_equivalent"] for row in diagnostics
        )
        / count,
        "argument_exact_match_accuracy": sum(
            row["argument_exact_match"] for row in diagnostics
        )
        / count,
        "argument_semantic_validity_rate": sum(
            row["argument_semantically_equivalent"] for row in diagnostics
        )
        / count,
        "argument_parse_rate": sum(
            row["argument_parse_success"] for row in diagnostics
        )
        / count,
        "plugin_execution_success_rate": sum(
            row["plugin_execution_success"] for row in diagnostics
        )
        / count,
        "final_answer_given_correct_tool": _conditional_rate(
            tool_correct_rows, "final_answer_correct"
        ),
        "final_answer_given_correct_tool_and_arguments": _conditional_rate(
            tool_and_args_rows, "final_answer_correct"
        ),
        "plugin_success_given_correct_tool": _conditional_rate(
            tool_correct_rows, "plugin_execution_success"
        ),
    }
    return metric, diagnostics


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    runs = discover_runs(
        args.routing_output_dir, ["tool_routing"], args.train_sizes
    )
    found_sizes = {train_size for _, train_size, _ in runs}
    missing = sorted(set(args.train_sizes) - found_sizes)
    if missing:
        raise FileNotFoundError(
            "Missing tool-routing adapters: "
            + ", ".join(f"train_{size}" for size in missing)
        )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    test_sets = {
        "seen": load_jsonl(args.data_dir / "test_seen.jsonl"),
        "unseen": load_jsonl(args.data_dir / "test_unseen.jsonl"),
    }
    direct = load_direct_accuracies(
        args.direct_predictions_csv, test_sets, args.train_sizes
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_metrics: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []

    for _, train_size, adapter_path in runs:
        base_model = _base_model_for(adapter_path, args.model_name)
        print(f"Evaluating tool routing, n={train_size}: {adapter_path}")
        tokenizer_source = (
            adapter_path
            if (adapter_path / "tokenizer_config.json").exists()
            else base_model
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_dtype = _model_dtype_for(adapter_path, args.model_dtype, torch)
        model_kwargs = {"torch_dtype": model_dtype} if model_dtype is not None else {}
        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        model = PeftModel.from_pretrained(base, adapter_path)
        model.to(device)
        model.eval()

        for split, rows in test_sets.items():
            completions = generate_predictions(
                model,
                tokenizer,
                rows,
                "tool_routing",
                args.batch_size,
                args.max_new_tokens,
                device,
            )
            metric, diagnostics = score_routing_split(
                rows,
                completions,
                train_size,
                split,
                direct[(train_size, split)],
            )
            all_metrics.append(metric)
            all_diagnostics.extend(diagnostics)
            print(
                f"  {split}: predicted_plugin_accuracy="
                f"{metric['predicted_plugin_accuracy']:.3f}, "
                f"plugin_success={metric['plugin_execution_success_rate']:.3f}"
            )

        del model, base, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(args.metrics_csv, METRIC_FIELDS, all_metrics)
    write_csv(args.predictions_csv, PREDICTION_FIELDS, all_diagnostics)
    print(f"Wrote hybrid metrics to {args.metrics_csv}")
    print(f"Wrote hybrid per-example diagnostics to {args.predictions_csv}")


if __name__ == "__main__":
    main()
