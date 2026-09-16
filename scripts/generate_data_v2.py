"""Generate taxonomy-v2 data without modifying the original dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from math_structure.data_v2 import generate_dataset_v2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "data_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=(50, 100, 200, 500))
    parser.add_argument("--seen-test-size", type=int, default=120)
    parser.add_argument("--unseen-test-size", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_dataset_v2(
        output_dir=args.output_dir,
        seed=args.seed,
        train_sizes=args.train_sizes,
        seen_test_size=args.seen_test_size,
        unseen_test_size=args.unseen_test_size,
    )
    print(f"Generated taxonomy-v2 data in {args.output_dir.resolve()}")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
