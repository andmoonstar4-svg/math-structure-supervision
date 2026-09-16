import json
from pathlib import Path

from math_structure.data import generate_dataset, verify_example


REQUIRED_FIELDS = {
    "id",
    "split",
    "template_split",
    "template_id",
    "structure",
    "tool",
    "problem",
    "answer",
    "baseline_target",
    "oracle_target",
    "structured_target",
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_generated_splits_are_verified_nested_and_template_disjoint(tmp_path: Path) -> None:
    generate_dataset(
        tmp_path,
        seed=7,
        train_sizes=(12, 24),
        seen_test_size=12,
        unseen_test_size=12,
    )

    train_12 = _read_jsonl(tmp_path / "train_12.jsonl")
    train_24 = _read_jsonl(tmp_path / "train_24.jsonl")
    seen = _read_jsonl(tmp_path / "test_seen.jsonl")
    unseen = _read_jsonl(tmp_path / "test_unseen.jsonl")

    assert (len(train_12), len(train_24), len(seen), len(unseen)) == (12, 24, 12, 12)
    assert {row["id"] for row in train_12} <= {row["id"] for row in train_24}

    train_templates = {row["template_id"] for row in train_24}
    seen_templates = {row["template_id"] for row in seen}
    unseen_templates = {row["template_id"] for row in unseen}
    assert seen_templates <= train_templates
    assert train_templates.isdisjoint(unseen_templates)

    for row in train_24 + seen + unseen:
        assert REQUIRED_FIELDS <= row.keys()
        assert verify_example(row)
        assert row["baseline_target"] == row["answer"]
        assert row["oracle_target"] == row["answer"]
        assert row["structure"] in row["structured_target"]
        assert row["tool"] in row["structured_target"]
        assert row["answer"] in row["structured_target"]


def test_generation_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    kwargs = dict(
        seed=123,
        train_sizes=(12,),
        seen_test_size=12,
        unseen_test_size=12,
    )

    generate_dataset(first, **kwargs)
    generate_dataset(second, **kwargs)

    for name in ("train_12.jsonl", "test_seen.jsonl", "test_unseen.jsonl", "metadata.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
