# Vision Transformer with prefix prompts.
#
# Adapted from DualPrompt-PyTorch (https://github.com/JH-LEE-KR/dualprompt-pytorch),
# Copyright (c) 2015-present, Facebook, Inc., with DualPrompt modifications by Jaeho Lee,
# itself based on the Vision Transformer of PyTorch Image Models
# (https://github.com/huggingface/pytorch-image-models), Copyright 2020 Ross Wightman.
# Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES).
#
# Changes: kept only the single shared prefix prompt used here (no prompt pool, keys or
# G-prompts) and removed unused layers and options. Modules are created and initialized in the
# original order, so a given seed produces the same prompt and head initialization.
from __future__ import annotations

from functools import partial

import timm
import torch
from timm.layers import Mlp, PatchEmbed, trunc_normal_
from torch import nn

from .checkpoint import load_state_dict


class PrefixAttention(nn.Module):
    """Multi-head self-attention whose keys and values are prefixed with learned prompt tokens."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        prompt = prompt.permute(1, 0, 3, 2, 4).contiguous()
        k = torch.cat([prompt[0], k], dim=2)
        v = torch.cat([prompt[1], v], dim=2)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, norm_layer) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PrefixAttention(dim, num_heads)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * 4, act_layer=nn.GELU, drop=0.0)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), prompt)
        return x + self.mlp(self.norm2(x))


class PrefixViT(nn.Module):
    """ViT-B/16 with a prefix prompt of ``prompt_length`` key and value tokens in every block."""

    def __init__(
        self,
        num_classes: int,
        prompt_length: int = 20,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
    ) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, embed_dim) * 0.02)
        # One key and one value prefix per block: (depth, 2, length, heads, head_dim).
        self.prompt = nn.Parameter(torch.randn(depth, 2, prompt_length, num_heads, embed_dim // num_heads))
        nn.init.uniform_(self.prompt, -1, 1)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, norm_layer) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) + self.pos_embed
        prompts = self.prompt.unsqueeze(1).expand(-1, x.shape[0], -1, -1, -1, -1)
        for i, block in enumerate(self.blocks):
            x = block(x, prompts[i])
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x)[:, 0])


def build_prefix_vit(checkpoint: str, num_classes: int, prompt_length: int) -> tuple[nn.Module, list, list]:
    """Prefix-prompt ViT-B/16 (ImageNet-21k); returns the model, prompt and head parameters.

    DualPrompt first builds a frozen ViT-B/16 that serves as its query network. With a single
    shared prompt the query is never used, so the model is discarded, but it is still built to
    draw the same random numbers before the prompt and head are initialized.
    """
    timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes, drop_rate=0.0, drop_path_rate=0.0)
    model = PrefixViT(num_classes=num_classes, prompt_length=prompt_length)
    state = {k: v for k, v in load_state_dict(checkpoint).items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if sorted(missing) != ["head.bias", "head.weight", "prompt"] or unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: missing={missing}, unexpected={unexpected}")
    for name, p in model.named_parameters():
        p.requires_grad = name == "prompt" or name.startswith("head.")
    return model, [model.prompt], list(model.head.parameters())
