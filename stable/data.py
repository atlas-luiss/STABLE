"""Datasets, transforms and class-incremental task splits."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def lora_transforms(dataset: str, image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    """Train and evaluation transforms of the LoRA pipeline (ImageNet normalization)."""
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if dataset == "cifar100":
        train = [transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip()]
        test = [transforms.Resize(image_size)]
    else:
        train = [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        ]
        test = [transforms.Resize(image_size + 32), transforms.CenterCrop(image_size)]
    return (
        transforms.Compose(train + [transforms.ToTensor(), normalize]),
        transforms.Compose(test + [transforms.ToTensor(), normalize]),
    )


def prompt_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    """Train and evaluation transforms of the prompt pipeline (DualPrompt recipe, no normalization)."""
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.05, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )
    test = transforms.Compose(
        [
            transforms.Resize(int((256 / 224) * image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    return train, test


def load_split(dataset: str, root: Path, train: bool, transform) -> Dataset:
    """Load the train or test split. Every returned dataset has ``targets`` and ``classes``."""
    if dataset == "cifar100":
        return datasets.CIFAR100(str(root), train=train, transform=transform, download=False)
    folder = root / ("train" if train else "test")
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found; see the data preparation section of the README.")
    return datasets.ImageFolder(str(folder), transform=transform)


def check_same_classes(train: Dataset, test: Dataset) -> int:
    if list(train.classes) != list(test.classes):
        raise ValueError("Train and test splits have different class folders.")
    return len(train.classes)


def task_classes(num_classes: int, num_tasks: int) -> list[list[int]]:
    """Split class ids ``0..num_classes-1`` into ``num_tasks`` consecutive groups of equal size."""
    if num_classes % num_tasks != 0:
        raise ValueError(f"{num_classes} classes cannot be split into {num_tasks} equal tasks.")
    per_task = num_classes // num_tasks
    return [list(range(t * per_task, (t + 1) * per_task)) for t in range(num_tasks)]


def class_subset(dataset: Dataset, classes: list[int]) -> Subset:
    """Samples of ``dataset`` whose label is in ``classes``, in dataset order."""
    keep = set(int(c) for c in classes)
    return Subset(dataset, [i for i, y in enumerate(dataset.targets) if int(y) in keep])


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """DataLoader with pinned memory. Shuffling uses ``generator``, or the global RNG if it is None."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator if shuffle else None,
    )
