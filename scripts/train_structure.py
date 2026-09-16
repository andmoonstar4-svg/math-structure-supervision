"""Train structure/tool/answer LoRA adapters for each requested sample size."""

try:
    from .train_common import build_training_parser, train_setting
except ImportError:  # Allows `python scripts/train_structure.py`.
    from train_common import build_training_parser, train_setting


def main() -> None:
    parser = build_training_parser(
        "Train the structure-aware problem -> structure -> tool -> answer condition."
    )
    train_setting("structure_aware", parser.parse_args())


if __name__ == "__main__":
    main()

