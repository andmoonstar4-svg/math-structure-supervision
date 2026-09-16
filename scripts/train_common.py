"""Shared prompting, parsing, and LoRA training code.

The experiment deliberately uses ordinary JSONL files and the Hugging Face
Trainer directly. Keeping these pieces here makes the four training entry
points differ only in the prompt and supervision target they select.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_TRAIN_SIZES = (50, 100, 200, 500)
SETTINGS = ("baseline", "structure_aware", "oracle", "tool_routing")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file and fail with a useful line number."""

    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path}, line {line_number}")
            rows.append(row)
    return rows


def _required(example: dict[str, Any], key: str) -> str:
    if key not in example:
        raise KeyError(f"Dataset example is missing required field {key!r}")
    return str(example[key]).strip()


def make_prompt(example: dict[str, Any], setting: str) -> str:
    """Create the input shown to the model for one supervision setting."""

    problem = _required(example, "problem")
    if setting == "baseline":
        return (
            "Solve the algebra problem. Return only the final answer.\n"
            f"Problem: {problem}\n"
            "Answer:"
        )
    if setting == "structure_aware":
        return (
            "Solve the algebra problem. Return exactly three lines in this format:\n"
            "STRUCTURE: <structure>\n"
            "TOOL: <tool>\n"
            "ANSWER: <answer>\n"
            f"Problem: {problem}\n"
            "Response:\n"
        )
    if setting == "oracle":
        structure = _required(example, "structure")
        tool = _required(example, "tool")
        return (
            "Solve the algebra problem using the supplied ground-truth labels. "
            "Return only the final answer.\n"
            f"Problem: {problem}\n"
            f"Structure: {structure}\n"
            f"Tool: {tool}\n"
            "Answer:"
        )
    if setting == "tool_routing":
        return (
            "Route the algebra problem to a deterministic solver. Return exactly "
            "three lines. ARGUMENTS must be one valid compact JSON object and do "
            "not return an answer.\n"
            "STRUCTURE: <structure>\n"
            "TOOL: <tool>\n"
            "ARGUMENTS: <json object>\n"
            f"Problem: {problem}\n"
            "Response:\n"
        )
    raise ValueError(f"Unknown setting {setting!r}; expected one of {SETTINGS}")


def make_target(example: dict[str, Any], setting: str) -> str:
    """Select (or reconstruct) the desired completion from a dataset row."""

    if setting in {"baseline", "oracle"}:
        # Prefer the raw answer.  Some generated files keep a labeled
        # ``baseline_target`` ("ANSWER: ...") for inspection, while this prompt
        # already ends in "Answer:" and should train on the answer itself.
        fallback_key = "oracle_target" if setting == "oracle" else "baseline_target"
        target = example.get("answer", example.get(fallback_key))
        if target is None:
            raise KeyError(f"Dataset example needs 'answer' or {fallback_key!r}")
        target = str(target).strip()
        if "answer" not in example:
            target = re.sub(r"(?i)^\s*ANSWER\s*:\s*", "", target)
        return target

    if setting == "structure_aware":
        target = example.get("structured_target")
        if target is not None:
            return str(target).strip()
        structure = _required(example, "structure")
        tool = _required(example, "tool")
        answer = _required(example, "answer")
        return f"STRUCTURE: {structure}\nTOOL: {tool}\nANSWER: {answer}"

    if setting == "tool_routing":
        target = example.get("routing_target")
        if target is not None:
            return str(target).strip()
        structure = _required(example, "structure")
        tool = _required(example, "tool")
        if "tool_args" not in example or not isinstance(example["tool_args"], dict):
            raise KeyError("Routing examples need a JSON-object 'tool_args' field")
        arguments = json.dumps(
            example["tool_args"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"STRUCTURE: {structure}\nTOOL: {tool}\nARGUMENTS: {arguments}"

    raise ValueError(f"Unknown setting {setting!r}; expected one of {SETTINGS}")


_TAG_PATTERN = re.compile(
    r"(?im)^\s*(STRUCTURE|TOOL|ANSWER)\s*:\s*(.*?)\s*$"
)


def parse_prediction(text: str, setting: str) -> dict[str, str | None]:
    """Extract comparable fields from a generated completion.

    Parsing is intentionally modest: it accepts case-insensitive field labels,
    while the metric itself remains exact after whitespace normalization.
    """

    fields: dict[str, str | None] = {
        "answer": None,
        "structure": None,
        "tool": None,
    }
    tagged = {key.lower(): value.strip() for key, value in _TAG_PATTERN.findall(text)}

    if setting == "structure_aware":
        fields.update(tagged)
        return fields
    if setting not in {"baseline", "oracle"}:
        raise ValueError(f"Unknown setting {setting!r}; expected one of {SETTINGS}")

    if "answer" in tagged:
        fields["answer"] = tagged["answer"]
        return fields

    # Baseline generations are requested to contain only an answer.  Taking the
    # first non-empty line prevents an otherwise-correct answer from being
    # spoiled by later free-form text.
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            fields["answer"] = candidate
            break
    return fields


def parse_routing_prediction(text: str) -> dict[str, Any]:
    """Parse structure, tool, and the first JSON object after ARGUMENTS."""

    tagged = {key.lower(): value.strip() for key, value in _TAG_PATTERN.findall(text)}
    marker = re.search(r"(?im)^\s*ARGUMENTS\s*:\s*", text)
    tool_args: dict[str, Any] | None = None
    arguments_text = ""
    if marker:
        remainder = text[marker.end() :].lstrip()
        try:
            parsed, consumed = json.JSONDecoder().raw_decode(remainder)
            arguments_text = remainder[:consumed]
            if isinstance(parsed, dict):
                tool_args = parsed
        except json.JSONDecodeError:
            arguments_text = remainder.splitlines()[0].strip() if remainder else ""
    return {
        "structure": tagged.get("structure"),
        "tool": tagged.get("tool"),
        "tool_args": tool_args,
        "tool_args_text": arguments_text,
    }


def normalize_answer(value: Any) -> str:
    """Normalize presentation whitespace while retaining exact answer content."""

    if value is None:
        return ""
    text = str(value).strip()
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1]
    return re.sub(r"\s+", "", text)


def normalize_label(value: Any) -> str:
    """Normalize harmless case/spacing differences in structure and tool labels."""

    if value is None:
        return ""
    return re.sub(r"[\s-]+", "_", str(value).strip().lower())


class SupervisedDataset:
    """A tiny causal-LM dataset that masks prompt tokens in the loss."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        tokenizer: Any,
        setting: str,
        max_length: int,
    ) -> None:
        self.examples: list[dict[str, list[int]]] = []
        eos = tokenizer.eos_token or ""

        for row in rows:
            prompt_ids = tokenizer(
                make_prompt(row, setting), add_special_tokens=True
            )["input_ids"]
            target = make_target(row, setting)
            # Answer-only prompts end with "Answer:" and benefit from an
            # explicit separating space. The structured prompt ends in a newline.
            if setting in {"baseline", "oracle"}:
                target = " " + target
            target_ids = tokenizer(
                target + eos, add_special_tokens=False
            )["input_ids"]

            target_ids = target_ids[:max_length]
            prompt_budget = max_length - len(target_ids)
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
            input_ids = prompt_ids + target_ids
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": [-100] * len(prompt_ids) + target_ids,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


@dataclass
class CausalCollator:
    """Right-pad a batch and use -100 for padded labels."""

    pad_token_id: int

    def __call__(self, examples: Sequence[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(example["input_ids"]) for example in examples)

        def padded(key: str, pad_value: int) -> list[list[int]]:
            return [
                example[key] + [pad_value] * (width - len(example[key]))
                for example in examples
            ]

        return {
            "input_ids": torch.tensor(padded("input_ids", self.pad_token_id)),
            "attention_mask": torch.tensor(padded("attention_mask", 0)),
            "labels": torch.tensor(padded("labels", -100)),
        }


def infer_lora_targets(model: Any) -> list[str]:
    """Choose a small attention projection set for common causal LMs."""

    leaf_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    candidates = (
        ("q_proj", "v_proj"),       # Llama/Qwen/SmolLM families
        ("query_key_value",),        # Falcon/Bloom-style fused attention
        ("c_attn",),                 # GPT-2
        ("Wqkv",),                   # Some MPT-style implementations
    )
    for group in candidates:
        if all(name in leaf_names for name in group):
            return list(group)
    raise ValueError(
        "Could not infer LoRA attention modules. Pass --lora-target-modules "
        "with module leaf names from this model."
    )


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAIN_SIZES),
        help="Training sizes to run; each uses data/train_<size>.jsonl.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=None,
        help="Optional model-specific module leaf names (auto-detected by default).",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade some speed for lower activation memory on larger models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow reuse of a non-empty output directory.",
    )


def _prepare_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def train_setting(setting: str, args: argparse.Namespace) -> None:
    """Train one fresh LoRA adapter per requested training-set size."""

    if setting not in SETTINGS:
        raise ValueError(f"Unknown setting {setting!r}; expected one of {SETTINGS}")
    if args.fp16 and args.bf16:
        raise ValueError("Choose at most one of --fp16 and --bf16")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

    tokenizer = _prepare_tokenizer(args.model_name)
    for train_size in args.train_sizes:
        set_seed(args.seed)
        train_path = args.data_dir / f"train_{train_size}.jsonl"
        rows = load_jsonl(train_path)
        if len(rows) != train_size:
            raise ValueError(
                f"{train_path} contains {len(rows)} examples, expected {train_size}"
            )

        run_dir = args.output_dir / setting / f"train_{train_size}"
        if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"{run_dir} is not empty; pass --overwrite to train there again"
            )
        run_dir.mkdir(parents=True, exist_ok=True)

        model_kwargs: dict[str, Any] = {}
        if args.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif args.fp16:
            model_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
        model.config.use_cache = False
        target_modules = args.lora_target_modules or infer_lora_targets(model)
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        if args.gradient_checkpointing:
            # PEFT freezes the embeddings. Checkpointed transformer blocks need
            # their floating-point inputs to require gradients so LoRA weights
            # still receive a backward pass.
            model.enable_input_require_grads()
        model.print_trainable_parameters()

        dataset = SupervisedDataset(rows, tokenizer, setting, args.max_length)
        logging_steps = max(1, len(dataset) // max(1, args.batch_size * 5))
        training_args = TrainingArguments(
            output_dir=str(run_dir),
            overwrite_output_dir=args.overwrite,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=0.05,
            lr_scheduler_type="constant_with_warmup",
            logging_steps=logging_steps,
            save_strategy="no",
            report_to=[],
            seed=args.seed,
            data_seed=args.seed,
            fp16=args.fp16,
            bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=CausalCollator(tokenizer.pad_token_id),
        )
        train_result = trainer.train()
        model.config.use_cache = True
        model.save_pretrained(run_dir)
        tokenizer.save_pretrained(run_dir)

        metadata = {
            "setting": setting,
            "train_size": train_size,
            "base_model": args.model_name,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": target_modules,
            "fp16": args.fp16,
            "bf16": args.bf16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "train_loss": train_result.training_loss,
        }
        (run_dir / "run_config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Saved {setting} adapter for n={train_size} to {run_dir}")

        del trainer, model, dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_training_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_training_arguments(parser)
    return parser


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
