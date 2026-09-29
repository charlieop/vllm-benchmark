"""Command-line entry point for independent benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .runner import archive, evaluate, generate, infer, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="visionbench", description="Stage 1 image-to-image benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "generate", "evaluate", "archive", "infer"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, help="Name in configs/ (for example qwen) or YAML path")
        if name == "infer":
            command.add_argument("--image", required=True, type=Path)
            command.add_argument("--prompt", required=True)
            command.add_argument("--system-prompt")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "run":
        print(run(config))
    elif args.command == "generate":
        print(json.dumps(generate(config), indent=2))
    elif args.command == "evaluate":
        print(evaluate(config))
    elif args.command == "archive":
        archive(config)
    elif args.command == "infer":
        print(infer(config, args.image, args.prompt, args.system_prompt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
