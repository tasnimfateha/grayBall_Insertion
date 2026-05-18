import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

import UNet_MobileNet as unet
from GreyBallInsertionDataset import GreyBallInsertionDataset


# =========================
# Configuration
# =========================

RUN_NAME = "grey_ball_insertion"

BALL_CSV = "updated_Ball_data.csv"  # use ball_data_modified.csv if you did not run SAM yet
HOMOGRAPHY_CSV = "matching_results.csv"

SCENES_DIR = "scenes"
MASKED_TARGETS_DIR = "masked_crops"  # use scene_shots_masked if you did not run SAM yet

BATCH_SIZE = 16
NUM_WORKERS = 0
EPOCHS = 100
LEARNING_RATE = 1e-4

TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

CHECKPOINT_DIR = "checkpoints"
VISUALIZATION_DIR = "training_outputs"


# =========================
# Loss function
# =========================

def masked_l1_loss(output, target, mask):
    """
    Computes L1 loss only on the ball region.

    output: predicted RGB ball patch, shape [B, 3, H, W]
    target: real RGB ball patch, shape [B, 3, H, W]
    mask: binary ball mask, shape [B, 1, H, W]

    The mask is expanded to 3 channels so it can be applied to RGB images.
    """

    mask_3ch = mask.expand(-1, 3, -1, -1)

    output_masked = output * mask_3ch
    target_masked = target * mask_3ch

    loss = torch.abs(output_masked - target_masked).sum()
    normalizer = mask_3ch.sum().clamp(min=1)

    return loss / normalizer


# =========================
# Training function
# =========================

def train_model():
    Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
    Path(VISUALIZATION_DIR).mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # Dataset
    # -------------------------

    full_dataset = GreyBallInsertionDataset(
        ball_csv_path=BALL_CSV,
        homography_csv_path=HOMOGRAPHY_CSV,
        scenes_dir=SCENES_DIR,
        masked_targets_dir=MASKED_TARGETS_DIR,
        crop_size=256,
        global_size=224,
    )

    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Dataset loaded:")
    print(f"  Train samples: {train_size}")
    print(f"  Validation samples: {val_size}")

    # -------------------------
    # Model
    # -------------------------

    model = unet.UNetMobileNetV3()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(log_dir=f"runs/{RUN_NAME}")

    # -------------------------
    # Fixed samples for visualization
    # -------------------------

    fixed_train_batch = next(iter(train_loader))
    fixed_val_batch = next(iter(val_loader))

    fixed_global_scene, fixed_local_crop, fixed_target, fixed_mask = [
        tensor[:6].to(device) for tensor in fixed_train_batch
    ]

    fixed_val_global_scene, fixed_val_local_crop, fixed_val_target, fixed_val_mask = [
        tensor[:6].to(device) for tensor in fixed_val_batch
    ]

    save_image(
        fixed_target,
        os.path.join(VISUALIZATION_DIR, "train_ground_truth.jpg"),
        nrow=3,
        normalize=True,
    )

    save_image(
        fixed_val_target,
        os.path.join(VISUALIZATION_DIR, "val_ground_truth.jpg"),
        nrow=3,
        normalize=True,
    )

    # -------------------------
    # Training loop
    # -------------------------

    start_time = time.time()
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # =====================
        # Training
        # =====================

        model.train()
        train_loss = 0.0

        for global_scene, local_crop, target_ball, mask in tqdm(
            train_loader,
            desc="Training",
        ):
            global_scene = global_scene.to(device)
            local_crop = local_crop.to(device)
            target_ball = target_ball.to(device)
            mask = mask.to(device)

            prediction = model(local_crop, global_scene)

            loss = masked_l1_loss(
                output=prediction,
                target=target_ball,
                mask=mask,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # =====================
        # Validation
        # =====================

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for global_scene, local_crop, target_ball, mask in tqdm(
                val_loader,
                desc="Validation",
            ):
                global_scene = global_scene.to(device)
                local_crop = local_crop.to(device)
                target_ball = target_ball.to(device)
                mask = mask.to(device)

                prediction = model(local_crop, global_scene)

                loss = masked_l1_loss(
                    output=prediction,
                    target=target_ball,
                    mask=mask,
                )

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        epoch_time = time.time() - epoch_start_time

        print(
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Epoch Time: {epoch_time:.2f}s"
        )

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Time/epoch_seconds", epoch_time, epoch)

        # =====================
        # Save visual predictions
        # =====================

        if epoch in [0, 1, 2, 5, 10, 20, 50, 75, EPOCHS - 1]:
            model.eval()

            with torch.no_grad():
                train_prediction = model(fixed_local_crop, fixed_global_scene)
                val_prediction = model(fixed_val_local_crop, fixed_val_global_scene)

                train_prediction_masked = train_prediction * fixed_mask.expand(-1, 3, -1, -1)
                val_prediction_masked = val_prediction * fixed_val_mask.expand(-1, 3, -1, -1)

                save_image(
                    train_prediction_masked,
                    os.path.join(VISUALIZATION_DIR, f"train_prediction_epoch_{epoch + 1}.jpg"),
                    nrow=3,
                    normalize=True,
                )

                save_image(
                    val_prediction_masked,
                    os.path.join(VISUALIZATION_DIR, f"val_prediction_epoch_{epoch + 1}.jpg"),
                    nrow=3,
                    normalize=True,
                )

        # =====================
        # Save checkpoint
        # =====================

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"checkpoint_epoch_{epoch + 1}.pt",
        )

        torch.save(checkpoint, checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            best_checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                "best_model.pt",
            )

            torch.save(checkpoint, best_checkpoint_path)

            print(f"Saved new best model with Val Loss: {best_val_loss:.6f}")

    writer.close()

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time / 60:.2f} minutes.")


if __name__ == "__main__":
    train_model()