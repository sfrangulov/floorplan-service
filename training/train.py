"""SegFormer training script for floorplan segmentation."""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    INPUT_SIZE,
    LEARNING_RATE,
    MODEL_NAME,
    NUM_CLASSES,
    NUM_EPOCHS,
    WEIGHT_DECAY,
)
from dataset import FloorplanDataset, get_train_transform, get_val_transform
from loss import CombinedLoss


def compute_miou(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> float:
    """Compute mean Intersection-over-Union, skipping background (class 0).

    Args:
        preds: (N,) predicted class labels (flattened)
        targets: (N,) ground truth class labels (flattened)
        num_classes: total number of classes

    Returns:
        Mean IoU over foreground classes
    """
    ious = []
    for cls in range(1, num_classes):  # skip background
        pred_mask = preds == cls
        target_mask = targets == cls
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union == 0:
            # Class not present in predictions or targets; skip
            continue
        ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0
    return float(np.mean(ious))


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch.

    Returns:
        Average loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=images)
        logits = outputs.logits  # (B, C, H/4, W/4)

        # SegFormer outputs at 1/4 resolution; interpolate to mask size
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation.

    Returns:
        Tuple of (average loss, mIoU).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    for images, masks in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(pixel_values=images)
        logits = outputs.logits

        # Interpolate to mask size
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = criterion(logits, masks)
        total_loss += loss.item()
        num_batches += 1

        preds = logits.argmax(dim=1)  # (B, H, W)
        all_preds.append(preds.cpu().flatten())
        all_targets.append(masks.cpu().flatten())

    avg_loss = total_loss / max(num_batches, 1)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    miou = compute_miou(all_preds, all_targets, NUM_CLASSES)

    return avg_loss, miou


def main():
    parser = argparse.ArgumentParser(
        description="Train SegFormer for floorplan segmentation"
    )
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        required=True,
        help="Training data directories (each with images/ and masks/ subdirs)",
    )
    parser.add_argument(
        "--val-dirs",
        nargs="+",
        required=True,
        help="Validation data directories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate (default: %(default)s)",
    )
    args = parser.parse_args()

    # Device selection: cuda > mps > cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Model
    print(f"Loading model: {MODEL_NAME}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Datasets
    train_dataset = FloorplanDataset(
        args.data_dirs, transform=get_train_transform()
    )
    val_dataset = FloorplanDataset(
        args.val_dirs, transform=get_val_transform()
    )
    print(
        f"Training samples: {len(train_dataset)}, "
        f"Validation samples: {len(val_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Loss, optimizer, scheduler
    criterion = CombinedLoss()
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = PolynomialLR(
        optimizer, total_iters=args.epochs, power=1.0
    )

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    best_dir = os.path.join(args.output_dir, "best")
    latest_dir = os.path.join(args.output_dir, "latest")

    # Training loop with early stopping
    best_miou = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_miou = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val mIoU: {val_miou:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save latest
        model.save_pretrained(latest_dir)
        print(f"Saved latest model to {latest_dir}")

        # Save best by mIoU
        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            model.save_pretrained(best_dir)
            print(f"New best mIoU: {best_miou:.4f} - saved to {best_dir}")
        else:
            patience_counter += 1
            print(
                f"No improvement for {patience_counter}/"
                f"{EARLY_STOPPING_PATIENCE} epochs"
            )

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
