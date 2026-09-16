import json
from pathlib import Path

from math_structure.data_v2 import (
    analyze_taxonomy,
    generate_dataset_v2,
    verify_example,
)
from math_structure.tools import execute_tool


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_v2_is_verified_held_out_and_many_to_many(tmp_path: Path) -> None:
    generate_dataset_v2(
        tmp_path,
        seed=9,
        train_sizes=(24, 48),
        seen_test_size=24,
        unseen_test_size=24,
    )

    train_24 = _read_jsonl(tmp_path / "train_24.jsonl")
    train_48 = _read_jsonl(tmp_path / "train_48.jsonl")
    seen = _read_jsonl(tmp_path / "test_seen.jsonl")
    unseen = _read_jsonl(tmp_path / "test_unseen.jsonl")
    assert {row["id"] for row in train_24} <= {row["id"] for row in train_48}
    assert {row["template_id"] for row in train_48}.isdisjoint(
        {row["template_id"] for row in unseen}
    )

    for row in train_48 + seen + unseen:
        assert row["taxonomy_version"] == "v2"
        assert row["oracle_target"] == row["answer"]
        assert isinstance(row["tool_args"], dict)
        assert row["routing_target"].endswith(
            json.dumps(
                row["tool_args"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        assert execute_tool(row["tool"], row["tool_args"]) == row["answer"]
        assert row["verification"]["plugin_passed"] is True
        assert verify_example(row)

    mapping = analyze_taxonomy(train_48 + seen + unseen)
    assert len(mapping["structure_to_tools"]) == 6
    assert len(mapping["tool_to_structures"]) == 7
    assert mapping["has_structure_with_multiple_tools"] is True
    assert mapping["has_tool_with_multiple_structures"] is True
    assert mapping["is_many_to_many"] is True
    assert mapping["is_one_to_one"] is False
    assert mapping["warning"] is None

    saved_mapping = json.loads(
        (tmp_path / "label_mapping.json").read_text(encoding="utf-8")
    )
    assert saved_mapping == mapping


def test_v2_generation_is_deterministic(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    kwargs = dict(
        seed=123,
        train_sizes=(24,),
        seen_test_size=24,
        unseen_test_size=24,
    )
    generate_dataset_v2(first, **kwargs)
    generate_dataset_v2(second, **kwargs)

    for name in (
        "train_24.jsonl",
        "test_seen.jsonl",
        "test_unseen.jsonl",
        "label_mapping.json",
        "metadata.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
