"""Command-line interface for GTLM training."""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .trainer import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a GTLM MoE language model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_base.json",
        help="Path to a JSON training config.",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load and validate the config, then exit before starting training.",
    )
    return parser


def apply_cli_overrides(config, args: argparse.Namespace) -> None:
    if args.resume_checkpoint:
        config.runtime.resume_checkpoint = args.resume_checkpoint
    if args.compile_model:
        config.runtime.compile_model = True
    if args.dataset_path:
        config.data.dataset_path = args.dataset_path
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.tokenizer_name:
        config.data.tokenizer_name = args.tokenizer_name


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    apply_cli_overrides(config, args)
    config.validate()

    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))

    if args.dry_run:
        print("Config validation passed.")
        return

    train(config, args)
