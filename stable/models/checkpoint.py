"""Loading of the ViT-B/16 ImageNet-21k weights."""
from __future__ import annotations

from pathlib import Path

import torch


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a ``.safetensors`` or ``.pth`` state dict onto the CPU."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; see the checkpoint section of the README.")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    state = torch.load(str(path), map_location="cpu")
    return state.get("state_dict", state)
