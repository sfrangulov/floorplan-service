"""PyTorch dataset for floorplan segmentation."""

import os

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import INPUT_SIZE, NUM_CLASSES


class FloorplanDataset(Dataset):
    """Dataset that loads image/mask pairs from directories.

    Each data directory must contain:
      - images/  with .png files
      - masks/   with matching .png files (grayscale class indices)
    """

    def __init__(self, data_dirs: list[str], transform=None):
        self.transform = transform
        self.samples: list[tuple[str, str]] = []

        for data_dir in data_dirs:
            images_dir = os.path.join(data_dir, "images")
            masks_dir = os.path.join(data_dir, "masks")

            if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
                continue

            for fname in sorted(os.listdir(images_dir)):
                if not fname.lower().endswith(".png"):
                    continue
                img_path = os.path.join(images_dir, fname)
                mask_path = os.path.join(masks_dir, fname)
                if os.path.isfile(mask_path):
                    self.samples.append((img_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        # Load image (BGR -> RGB)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize to INPUT_SIZE if needed
        if image.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            image = cv2.resize(
                image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR
            )
        if mask.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            mask = cv2.resize(
                mask, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST
            )

        # Clamp mask values to valid class range
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        # Apply albumentations transform
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensors: image CHW float [0,1], mask long
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor


def get_train_transform() -> A.Compose:
    """Return augmentation pipeline for training."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05,
                p=0.5,
            ),
            A.GaussNoise(var_limit=(5, 25), p=0.3),
            A.ElasticTransform(alpha=20, sigma=5, p=0.2),
        ]
    )


def get_val_transform():
    """Return augmentation pipeline for validation (no augmentations)."""
    return None
