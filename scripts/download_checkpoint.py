#!/usr/bin/env python3
"""Download the ViT-B/16 ImageNet-21k weights used in the paper and check their SHA-256.

    python scripts/download_checkpoint.py [--out checkpoints/vit_base_patch16_224.augreg_in21k.safetensors]

The file is ``model.safetensors`` of timm's ``vit_base_patch16_224.augreg_in21k`` (the AugReg
ImageNet-21k ViT-B/16 of Steiner et al., 2022) from the Hugging Face Hub.
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k/resolve/main/model.safetensors"
SHA256 = "2d31bb5597c720912f8ea5e4f1de73909a6911798d78ae5d1b9d63a955fed049"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/vit_base_patch16_224.augreg_in21k.safetensors"))
    args = parser.parse_args()
    if not args.out.exists():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, args.out)
    digest = sha256_file(args.out)
    if digest != SHA256:
        raise SystemExit(f"SHA-256 mismatch for {args.out}: {digest}")
    print(f"{args.out}: SHA-256 verified.")


if __name__ == "__main__":
    main()
