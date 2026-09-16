"""Train answer-only LoRA adapters with gold structure and tool in the prompt."""

try:
    from .train_common import build_training_parser, train_setting
except ImportError:  # Allows `python scripts/train_oracle.py`.
    from train_common import build_training_parser, train_setting


def main() -> None:
    parser = build_training_parser(
        "Train the oracle condition with gold structure/tool labels in the prompt."
    )
    train_setting("oracle", parser.parse_args())


if __name__ == "__main__":
    main()
