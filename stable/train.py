"""Run one class-incremental stream.

Example:
    python -m stable.train --config configs/table1/lora_cifar100.yaml --seed 42
    python -m stable.train --config configs/table1/lora_cifar100.yaml --set max_tasks=2 epochs=1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config, parse_overrides
from .pipelines import lora, prompts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train ViT-B/16 with STABLE on a class-incremental stream.")
    parser.add_argument("--config", required=True, help="YAML config, e.g. configs/table1/lora_cifar100.yaml")
    parser.add_argument("--seed", type=int, help="random seed (default: the config value)")
    parser.add_argument("--data-dir", help="directory with the prepared datasets (default: data)")
    parser.add_argument("--checkpoint", help="ViT-B/16 ImageNet-21k weights (.safetensors or .pth)")
    parser.add_argument("--output-dir", help="results go to <output-dir>/<config name>/seed<seed>")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config keys")
    args = parser.parse_args(argv)

    overrides = parse_overrides(args.set)
    for key in ("seed", "data_dir", "checkpoint", "output_dir"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    cfg = load_config(args.config, overrides)

    run_dir = Path(cfg.output_dir).expanduser() / Path(args.config).stem / f"seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "log.txt", mode="w")],
    )
    logging.getLogger("stable").info(f"Config: {cfg.to_dict()}")
    (lora if cfg.method == "lora" else prompts).run(cfg, run_dir)


if __name__ == "__main__":
    main()
