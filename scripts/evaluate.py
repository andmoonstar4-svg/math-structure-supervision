"""Evaluate every trained adapter and write aggregate and per-example CSVs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

try:
    from .train_common import (
        DEFAULT_MODEL,
        SETTINGS,
        batched,
        load_jsonl,
        make_prompt,
        normalize_answer,
        normalize_label,
        parse_prediction,
    )
except ImportError:  # Allows `python scripts/evaluate.py`.
    from train_common import (
        DEFAULT_MODEL,
        SETTINGS,
        batched,
        load_jsonl,
        make_prompt,
        normalize_answer,
        normalize_label,
        parse_prediction,
    )


METRIC_FIELDS = [
    "setting",
    "train_size",
    "template_split",
    "n",
    "answer_correct",
    "exact_answer_accuracy",
    "structure_correct",
    "structure_accuracy",
    "tool_correct",
    "tool_accuracy",
    "base_model",
    "adapter_path",
]

PREDICTION_FIELDS = [
    "id",
    "setting",
    "train_size",
    "template_split",
    "problem",
    "gold_answer",
    "predicted_answer",
    "answer_correct",
    "gold_structure",
    "predicted_structure",
    "structure_correct",
    "gold_tool",
    "predicted_tool",
    "tool_correct",
    "raw_prediction",
]

CONDITIONAL_FIELDS = [
    "train_size",
    "split",
    "n",
    "correct_structure_n",
    "correct_tool_n",
    "correct_structure_and_tool_n",
    "structure_or_tool_incorrect_n",
    "answer_given_correct_structure",
    "answer_given_correct_tool",
    "answer_given_correct_structure_and_tool",
    "answer_given_structure_or_tool_incorrect",
]

COMPARISON_FIELDS = [
    "train_size",
    "split",
    "baseline_answer_accuracy",
    "structure_answer_accuracy",
    "oracle_answer_accuracy",
    "structure_accuracy",
    "tool_accuracy",
    "answer_given_correct_structure",
    "answer_given_correct_tool",
    "answer_given_correct_structure_and_tool",
    "answer_given_structure_or_tool_incorrect",
]

STRUCTURE_CONFUSION_FIELDS = [
    "train_size",
    "split",
    "gold_structure",
    "predicted_structure",
    "count",
]

TOOL_CONFUSION_FIELDS = [
    "train_size",
    "split",
    "gold_tool",
    "predicted_tool",
    "count",
]

DIRECT_SETTINGS = ("baseline", "structure_aware", "oracle")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA adapters on seen- and unseen-template test data."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--results-csv", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("results/predictions.csv"),
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("results/comparison.csv"),
    )
    parser.add_argument(
        "--conditional-csv",
        type=Path,
        default=Path("results/conditional_metrics.csv"),
    )
    parser.add_argument(
        "--label-mapping-json",
        type=Path,
        default=Path("results/label_mapping.json"),
    )
    parser.add_argument(
        "--structure-confusion-csv",
        type=Path,
        default=Path("results/structure_confusion.csv"),
    )
    parser.add_argument(
        "--tool-confusion-csv",
        type=Path,
        default=Path("results/tool_confusion.csv"),
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=SETTINGS,
        default=list(DIRECT_SETTINGS),
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Optional sizes to evaluate. By default, discover all saved adapters.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=f"Override adapter metadata (fallback: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--model-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Base-model load dtype. Auto reuses the adapter training precision.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser


def discover_runs(
    output_dir: Path,
    settings: list[str],
    train_sizes: list[int] | None,
) -> list[tuple[str, int, Path]]:
    """Find complete PEFT adapter directories in deterministic order."""

    runs: list[tuple[str, int, Path]] = []
    wanted_sizes = set(train_sizes) if train_sizes is not None else None
    for setting in settings:
        setting_dir = output_dir / setting
        if not setting_dir.exists():
            continue
        for path in setting_dir.glob("train_*"):
            try:
                train_size = int(path.name.removeprefix("train_"))
            except ValueError:
                continue
            if wanted_sizes is not None and train_size not in wanted_sizes:
                continue
            if (path / "adapter_config.json").exists():
                runs.append((setting, train_size, path))
    return sorted(runs, key=lambda item: (item[1], item[0]))


def _base_model_for(adapter_path: Path, override: str | None) -> str:
    if override:
        return override
    metadata_path = adapter_path / "run_config.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("base_model"):
            return str(metadata["base_model"])
    # PEFT also records the base model, but this fallback keeps evaluation useful
    # if the small experiment metadata was deleted.
    adapter_config = json.loads(
        (adapter_path / "adapter_config.json").read_text(encoding="utf-8")
    )
    return str(adapter_config.get("base_model_name_or_path") or DEFAULT_MODEL)


def _model_dtype_for(adapter_path: Path, requested: str, torch: Any) -> Any:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    metadata_path = adapter_path / "run_config.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("bf16"):
            return torch.bfloat16
        if metadata.get("fp16"):
            return torch.float16
    return None


def generate_predictions(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    setting: str,
    batch_size: int,
    max_new_tokens: int,
    device: Any,
) -> list[str]:
    """Greedily generate only the completion portion for each prompt."""

    import torch

    outputs: list[str] = []
    prompts = [make_prompt(row, setting) for row in rows]
    tokenizer.padding_side = "left"
    for prompt_batch in batched(prompts, batch_size):
        encoded = tokenizer(
            list(prompt_batch),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = generated[:, input_width:]
        outputs.extend(tokenizer.batch_decode(completion_ids, skip_special_tokens=True))
    return outputs


def score_split(
    rows: list[dict[str, Any]],
    completions: list[str],
    setting: str,
    train_size: int,
    template_split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(rows) != len(completions):
        raise ValueError("Number of generations does not match number of test rows")

    predictions: list[dict[str, Any]] = []
    answer_correct = 0
    structure_correct = 0
    tool_correct = 0

    for index, (row, completion) in enumerate(zip(rows, completions)):
        parsed = parse_prediction(completion, setting)
        is_answer_correct = normalize_answer(parsed["answer"]) == normalize_answer(
            row.get("answer")
        )
        answer_correct += int(is_answer_correct)

        if setting == "structure_aware":
            is_structure_correct: bool | None = normalize_label(
                parsed["structure"]
            ) == normalize_label(row.get("structure"))
            is_tool_correct: bool | None = normalize_label(parsed["tool"]) == normalize_label(
                row.get("tool")
            )
            structure_correct += int(is_structure_correct)
            tool_correct += int(is_tool_correct)
        else:
            is_structure_correct = None
            is_tool_correct = None

        predictions.append(
            {
                "id": row.get("id", index),
                "setting": setting,
                "train_size": train_size,
                "template_split": template_split,
                "problem": row.get("problem", ""),
                "gold_answer": row.get("answer", ""),
                "predicted_answer": parsed["answer"] or "",
                "answer_correct": is_answer_correct,
                "gold_structure": row.get("structure", ""),
                "predicted_structure": parsed["structure"] or "",
                "structure_correct": is_structure_correct,
                "gold_tool": row.get("tool", ""),
                "predicted_tool": parsed["tool"] or "",
                "tool_correct": is_tool_correct,
                "raw_prediction": completion,
            }
        )

    count = len(rows)
    metric = {
        "setting": setting,
        "train_size": train_size,
        "template_split": template_split,
        "n": count,
        "answer_correct": answer_correct,
        "exact_answer_accuracy": answer_correct / count if count else 0.0,
        # Blank values make clear that the answer-only baseline was never asked
        # to predict these labels; treating them as zero would be misleading.
        "structure_correct": structure_correct if setting == "structure_aware" else None,
        "structure_accuracy": (
            structure_correct / count if count and setting == "structure_aware" else None
        ),
        "tool_correct": tool_correct if setting == "structure_aware" else None,
        "tool_accuracy": tool_correct / count if count and setting == "structure_aware" else None,
    }
    return metric, predictions


def conditional_execution_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure answer accuracy within structure/tool correctness subsets."""

    usable = [
        row
        for row in predictions
        if row.get("structure_correct") is not None
        and row.get("tool_correct") is not None
    ]

    def conditional_accuracy(selected: list[dict[str, Any]]) -> float | None:
        if not selected:
            return None
        return sum(bool(row.get("answer_correct")) for row in selected) / len(selected)

    correct_structure = [row for row in usable if row["structure_correct"]]
    correct_tool = [row for row in usable if row["tool_correct"]]
    correct_both = [
        row for row in usable if row["structure_correct"] and row["tool_correct"]
    ]
    either_incorrect = [
        row
        for row in usable
        if not (row["structure_correct"] and row["tool_correct"])
    ]
    return {
        "n": len(usable),
        "correct_structure_n": len(correct_structure),
        "correct_tool_n": len(correct_tool),
        "correct_structure_and_tool_n": len(correct_both),
        "structure_or_tool_incorrect_n": len(either_incorrect),
        "answer_given_correct_structure": conditional_accuracy(correct_structure),
        "answer_given_correct_tool": conditional_accuracy(correct_tool),
        "answer_given_correct_structure_and_tool": conditional_accuracy(correct_both),
        "answer_given_structure_or_tool_incorrect": conditional_accuracy(
            either_incorrect
        ),
    }


def analyze_label_mapping(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether structure and tool labels uniquely determine each other."""

    structure_to_tools: dict[str, set[str]] = {}
    tool_to_structures: dict[str, set[str]] = {}
    for row in rows:
        structure = str(row["structure"])
        tool = str(row["tool"])
        structure_to_tools.setdefault(structure, set()).add(tool)
        tool_to_structures.setdefault(tool, set()).add(structure)

    structure_has_choices = any(
        len(tools) > 1 for tools in structure_to_tools.values()
    )
    tool_is_reused = any(
        len(structures) > 1 for structures in tool_to_structures.values()
    )
    one_to_one = bool(structure_to_tools) and all(
        len(tools) == 1 for tools in structure_to_tools.values()
    ) and all(len(structures) == 1 for structures in tool_to_structures.values())
    warning = None
    if one_to_one:
        warning = (
            "Every structure maps to exactly one tool and every tool maps to exactly "
            "one structure; the two auxiliary labels are effectively redundant in "
            "this dataset."
        )
    return {
        "structure_to_tools": {
            structure: sorted(tools)
            for structure, tools in sorted(structure_to_tools.items())
        },
        "tool_to_structures": {
            tool: sorted(structures)
            for tool, structures in sorted(tool_to_structures.items())
        },
        "has_structure_with_multiple_tools": structure_has_choices,
        "has_tool_with_multiple_structures": tool_is_reused,
        "is_many_to_many": structure_has_choices and tool_is_reused,
        "is_one_to_one": one_to_one,
        "warning": warning,
    }


def load_complete_dataset(data_dir: Path) -> list[dict[str, Any]]:
    """Load the largest nested train split and both test splits once each."""

    train_files: list[tuple[int, Path]] = []
    for path in data_dir.glob("train_*.jsonl"):
        try:
            train_files.append((int(path.stem.removeprefix("train_")), path))
        except ValueError:
            continue
    if not train_files:
        raise FileNotFoundError(f"No train_<N>.jsonl files found in {data_dir}")
    largest_train = max(train_files, key=lambda item: item[0])[1]
    rows = load_jsonl(largest_train)
    rows.extend(load_jsonl(data_dir / "test_seen.jsonl"))
    rows.extend(load_jsonl(data_dir / "test_unseen.jsonl"))
    return rows


def build_comparison_rows(
    metrics: list[dict[str, Any]],
    conditional_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one wide row per training size and template split."""

    rows_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for metric in metrics:
        key = (int(metric["train_size"]), str(metric["template_split"]))
        row = rows_by_key.setdefault(
            key,
            {field: None for field in COMPARISON_FIELDS},
        )
        row["train_size"], row["split"] = key
        setting = metric["setting"]
        if setting == "baseline":
            row["baseline_answer_accuracy"] = metric["exact_answer_accuracy"]
        elif setting == "structure_aware":
            row["structure_answer_accuracy"] = metric["exact_answer_accuracy"]
            row["structure_accuracy"] = metric["structure_accuracy"]
            row["tool_accuracy"] = metric["tool_accuracy"]
        elif setting == "oracle":
            row["oracle_answer_accuracy"] = metric["exact_answer_accuracy"]

    for conditional in conditional_rows:
        key = (int(conditional["train_size"]), str(conditional["split"]))
        row = rows_by_key.setdefault(
            key,
            {field: None for field in COMPARISON_FIELDS},
        )
        row["train_size"], row["split"] = key
        for field in COMPARISON_FIELDS:
            if field.startswith("answer_given_"):
                row[field] = conditional.get(field)

    split_order = {"seen": 0, "unseen": 1}
    return [
        rows_by_key[key]
        for key in sorted(
            rows_by_key,
            key=lambda item: (item[0], split_order.get(item[1], 99), item[1]),
        )
    ]


def build_confusion_rows(
    predictions: list[dict[str, Any]],
    gold_field: str,
    predicted_field: str,
) -> list[dict[str, Any]]:
    """Aggregate structure-aware label predictions into a tidy confusion CSV."""

    counts: dict[tuple[int, str, str, str], int] = {}
    for row in predictions:
        if row.get("setting") != "structure_aware":
            continue
        key = (
            int(row["train_size"]),
            str(row["template_split"]),
            str(row.get(gold_field) or "<missing>"),
            str(row.get(predicted_field) or "<missing>"),
        )
        counts[key] = counts.get(key, 0) + 1

    return [
        {
            "train_size": train_size,
            "split": split,
            gold_field: gold,
            predicted_field: predicted,
            "count": count,
        }
        for (train_size, split, gold, predicted), count in sorted(counts.items())
    ]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    runs = discover_runs(args.output_dir, args.settings, args.train_sizes)
    if not runs:
        raise FileNotFoundError(
            f"No trained adapters found below {args.output_dir}. Run the training scripts first."
        )
    if args.train_sizes is not None:
        found = {(setting, train_size) for setting, train_size, _ in runs}
        expected = {
            (setting, train_size)
            for setting in args.settings
            for train_size in args.train_sizes
        }
        missing = sorted(expected - found, key=lambda item: (item[1], item[0]))
        if missing:
            missing_text = ", ".join(
                f"{setting}/train_{train_size}" for setting, train_size in missing
            )
            raise FileNotFoundError(
                f"Requested adapters are missing below {args.output_dir}: {missing_text}"
            )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_sets = {
        "seen": load_jsonl(args.data_dir / "test_seen.jsonl"),
        "unseen": load_jsonl(args.data_dir / "test_unseen.jsonl"),
    }
    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []

    label_mapping = analyze_label_mapping(load_complete_dataset(args.data_dir))
    write_json(args.label_mapping_json, label_mapping)
    if label_mapping["warning"]:
        print(f"WARNING: {label_mapping['warning']}")

    for setting, train_size, adapter_path in runs:
        base_model = _base_model_for(adapter_path, args.model_name)
        print(f"Evaluating {setting}, n={train_size}: {adapter_path}")
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

        for template_split, rows in test_sets.items():
            completions = generate_predictions(
                model,
                tokenizer,
                rows,
                setting,
                args.batch_size,
                args.max_new_tokens,
                device,
            )
            metric, predictions = score_split(
                rows, completions, setting, train_size, template_split
            )
            metric["base_model"] = base_model
            metric["adapter_path"] = str(adapter_path)
            all_metrics.append(metric)
            all_predictions.extend(predictions)
            if setting == "structure_aware":
                conditional = conditional_execution_metrics(predictions)
                conditional.update(
                    {"train_size": train_size, "split": template_split}
                )
                conditional_rows.append(conditional)
            print(
                "  "
                f"{template_split}: exact_answer_accuracy="
                f"{metric['exact_answer_accuracy']:.3f}"
            )

        del model, base, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(args.results_csv, METRIC_FIELDS, all_metrics)
    write_csv(args.predictions_csv, PREDICTION_FIELDS, all_predictions)
    write_csv(args.conditional_csv, CONDITIONAL_FIELDS, conditional_rows)
    comparison_rows = build_comparison_rows(all_metrics, conditional_rows)
    write_csv(args.comparison_csv, COMPARISON_FIELDS, comparison_rows)
    structure_confusion = build_confusion_rows(
        all_predictions, "gold_structure", "predicted_structure"
    )
    tool_confusion = build_confusion_rows(
        all_predictions, "gold_tool", "predicted_tool"
    )
    write_csv(
        args.structure_confusion_csv,
        STRUCTURE_CONFUSION_FIELDS,
        structure_confusion,
    )
    write_csv(args.tool_confusion_csv, TOOL_CONFUSION_FIELDS, tool_confusion)
    print(f"Wrote aggregate metrics to {args.results_csv}")
    print(f"Wrote per-example predictions to {args.predictions_csv}")
    print(f"Wrote conditional execution metrics to {args.conditional_csv}")
    print(f"Wrote wide comparison table to {args.comparison_csv}")
    print(f"Wrote label mapping analysis to {args.label_mapping_json}")
    print(f"Wrote structure confusion counts to {args.structure_confusion_csv}")
    print(f"Wrote tool confusion counts to {args.tool_confusion_csv}")


if __name__ == "__main__":
    main()
