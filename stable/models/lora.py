"""LoRA adapters on the attention projections of a timm ViT."""
from __future__ import annotations

import timm
import torch
from torch import nn

from .checkpoint import load_state_dict


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank update ``scaling * A @ B``.

    ``A`` (out x rank) is drawn from N(0, init_std^2) and ``B`` (rank x in) starts at zero, so the
    adapted layer equals the pretrained one at initialization.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, init_std: float = 1e-3) -> None:
        super().__init__()
        self.base = base
        self.scaling = float(alpha) / float(rank)
        self.A = nn.Parameter(torch.zeros(base.out_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, base.in_features))
        for p in self.base.parameters():
            p.requires_grad = False
        nn.init.normal_(self.A, mean=0.0, std=init_std)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * ((x @ self.B.t()) @ self.A.t())


def inject_lora(module: nn.Module, rank: int, alpha: float, prefix: str = "") -> None:
    """Wrap every biased ``nn.Linear`` inside the attention blocks (``qkv`` and ``proj``)."""
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and child.bias is not None and "attn" in full_name:
            setattr(module, name, LoRALinear(child, rank, alpha))
        else:
            inject_lora(child, rank, alpha, full_name)


def build_lora_vit(
    checkpoint: str, num_classes: int, rank: int, alpha: float
) -> tuple[nn.Module, list[nn.Parameter], list[nn.Parameter]]:
    """ViT-B/16 (ImageNet-21k) with a new linear head and LoRA on the attention projections.

    Returns the model, the LoRA parameters and the head parameters; everything else is frozen.
    """
    model = timm.create_model("vit_base_patch16_224.augreg_in21k", pretrained=False)
    model.load_state_dict(load_state_dict(checkpoint))
    model.reset_classifier(num_classes=num_classes)
    inject_lora(model, rank, alpha)
    for p in model.parameters():
        p.requires_grad = False
    lora_ids = set()
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.A.requires_grad = True
            m.B.requires_grad = True
            lora_ids.update((id(m.A), id(m.B)))
    for p in model.head.parameters():
        p.requires_grad = True
    lora_params = [p for p in model.parameters() if id(p) in lora_ids]
    head_params = list(model.head.parameters())
    return model, lora_params, head_params
