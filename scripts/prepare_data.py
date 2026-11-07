#!/usr/bin/env python3
"""Download and arrange the datasets of the Table 1 streams.

    python scripts/prepare_data.py cifar100
    python scripts/prepare_data.py imagenet-r [--tar imagenet-r.tar]
    python scripts/prepare_data.py imagenet-a [--tar imagenet-a.tar]
    python scripts/prepare_data.py imagenet-r --verify-only

ImageNet-R and ImageNet-A-100 are extracted into ``<data-dir>/<name>/{train,test}/<wnid>/`` from
the official archives, following the file lists in ``splits/``. The checksum of each list and a
SHA-256 digest of the extracted images are checked, so the data are identical to those used for
the paper. ``--verify-only`` checks an existing folder without extracting.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    "imagenet-r": {
        "url": "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
        "folder": "imagenet-r",
        "list": "imagenet_r.txt",
        "list_sha256": "7ab699862fba8c8bbc092ff482b136caebb9f92b84f8876bd1d8836be8288449",
        "content_sha256": "1ea5b75dd35ee4ff16e3e37b5dd29f684ad8174f7990bc789cd86623b931b08f",
    },
    "imagenet-a": {
        "url": "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar",
        "folder": "imagenet-a-top100",
        "list": "imagenet_a100.txt",
        "list_sha256": "384e2b227f183aaf30d02a998321f9c538668406f52452a1479e82ad1c80d547",
        "content_sha256": "ff2154aebc43c6254da4ca8a0c518cfe9e2fe99c1700636488e2f808cad4d0b9",
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_split_list(spec: dict) -> dict[str, str]:
    """Map ``<wnid>/<file>`` to ``train`` or ``test`` after checking the list's checksum."""
    path = ROOT / "splits" / spec["list"]
    data = path.read_bytes()
    if sha256_bytes(data) != spec["list_sha256"]:
        raise SystemExit(f"Checksum mismatch for {path}; the file list was modified.")
    mapping = {}
    for line in data.decode().splitlines():
        split, key = line.split("/", 1)
        mapping[key] = split
    return mapping


def extract(tar_path: Path, mapping: dict[str, str], out: Path) -> None:
    found = 0
    with tarfile.open(tar_path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            key = "/".join(member.name.split("/")[-2:])
            split = mapping.get(key)
            if split is None:
                continue
            target = out / split / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())
            found += 1
    if found != len(mapping):
        raise SystemExit(f"{tar_path} contains {found} of the {len(mapping)} listed images.")


def content_digest(out: Path, mapping: dict[str, str]) -> str:
    """SHA-256 over the sorted lines ``<split>/<wnid>/<file><TAB><sha256 of the file>``."""
    rows = []
    for split in ("train", "test"):
        for wnid in sorted(os.listdir(out / split)):
            folder = out / split / wnid
            if not folder.is_dir():
                continue
            for name in sorted(os.listdir(folder)):
                key = f"{wnid}/{name}"
                if mapping.get(key) != split:
                    raise SystemExit(f"Unexpected file {folder / name}.")
                rows.append(f"{split}/{key}\t{sha256_bytes((folder / name).read_bytes())}\n")
    if len(rows) != len(mapping):
        raise SystemExit(f"{out} holds {len(rows)} of the {len(mapping)} listed images.")
    return sha256_bytes("".join(sorted(rows)).encode())


def prepare_split(name: str, data_dir: Path, tar_path: Path | None, verify_only: bool) -> None:
    spec = SPLITS[name]
    out = data_dir / spec["folder"]
    mapping = read_split_list(spec)
    if not verify_only:
        if tar_path is None:
            tar_path = data_dir / Path(spec["url"]).name
            if not tar_path.exists():
                print(f"Downloading {spec['url']} to {tar_path}")
                tar_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(spec["url"], tar_path)
        print(f"Extracting {len(mapping)} images from {tar_path} into {out}")
        extract(tar_path, mapping, out)
    digest = content_digest(out, mapping)
    if digest != spec["content_sha256"]:
        raise SystemExit(f"Content digest mismatch for {out}: {digest}")
    counts = {s: sum(1 for v in mapping.values() if v == s) for s in ("train", "test")}
    print(f"{out}: {counts['train']} train and {counts['test']} test images, content verified.")


def prepare_cifar100(data_dir: Path) -> None:
    from torchvision.datasets import CIFAR100

    root = data_dir / "cifar100"
    for train in (True, False):
        CIFAR100(str(root), train=train, download=True)
    print(f"{root}: CIFAR-100 ready (torchvision checks the MD5 checksums).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["cifar100", "imagenet-r", "imagenet-a"])
    parser.add_argument("--data-dir", default="data", help="output directory (default: data)")
    parser.add_argument("--tar", type=Path, help="official archive, if already downloaded")
    parser.add_argument("--verify-only", action="store_true", help="only verify an existing folder")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser()
    if args.dataset == "cifar100":
        prepare_cifar100(data_dir)
    else:
        prepare_split(args.dataset, data_dir, args.tar, args.verify_only)


if __name__ == "__main__":
    main()
