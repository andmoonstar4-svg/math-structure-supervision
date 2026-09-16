"""Train answer-only LoRA adapters for each requested sample size."""

try:
    from .train_common import build_training_parser, train_setting
except ImportError:  # Allows `python scripts/train_baseline.py`.
    from train_common import build_training_parser, train_setting


def main() -> None:
    parser = build_training_parser(
        "Train the baseline problem -> answer condition with LoRA."
    )
    train_setting("baseline", parser.parse_args())


if __name__ == "__main__":
    main()

