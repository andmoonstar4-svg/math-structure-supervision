"""Train LoRA adapters to predict structure, tool, and structured tool arguments."""

try:
    from .train_common import build_training_parser, train_setting
except ImportError:  # Allows `python scripts/train_tool_routing.py`.
    from train_common import build_training_parser, train_setting


def main() -> None:
    parser = build_training_parser(
        "Train the problem -> structure/tool/tool_args routing condition with LoRA."
    )
    train_setting("tool_routing", parser.parse_args())


if __name__ == "__main__":
    main()
