"""Utilities for the small structure-supervision experiment."""

from .data import (
    STRUCTURE_SPECS,
    generate_dataset,
    generate_examples,
    read_jsonl,
    verify_example,
)

__all__ = [
    "STRUCTURE_SPECS",
    "generate_dataset",
    "generate_examples",
    "read_jsonl",
    "verify_example",
]
