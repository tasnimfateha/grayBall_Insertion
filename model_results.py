import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

import UNet_MobileNet as unet
from GreyBallInsertionDataset import GreyBallInsertionDataset


# Configuration

BALL_CSV = "Updated_Ball_Data.csv"
HOMOGRAPHY_CSV = "matching_results.csv"

SCENES_DIR = "scenes"
MASKED_TARGETS_DIR = "Masked_Crops"
CHECKPOINT_PATH = "checkpoints/best_model.pt"
OUTPUT_DIR = "final_visualizations"

TRAIN_SPLIT = 0.8
RANDOM_SEED = 42
BATCH_SIZE = 6
NUM_WORKERS = 0


# Helper functions

def save_sample_panel(local_crop, gt_ball, pred_ball, gt_overlay, pred_overlay, save_path):

    panel = torch.stack(
        [local_crop, gt_ball, pred_ball, gt_overlay, pred_overlay],
        dim=0
    )

    save_image(panel, save_path, nrow=5, normalize=True)


def paste_ball_into_global_scene(global_scene, pred_ball, mask, crop_box):
    """
    Paste the predicted ball crop back into the full global scene.

    global_scene: [3, H, W]
    pred_ball: [3, 256, 256]
    mask: [1, 256, 256]
    crop_box: [x1, y1, x2, y2]
    """

    full_overlay = global_scene.clone()

    x1, y1, x2, y2 = crop_box.tolist()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    mask_3ch = mask.expand(3, -1, -1)

    scene_crop = full_overlay[:, y1:y2, x1:x2]

    blended_crop = scene_crop * (1 - mask_3ch) + pred_ball * mask_3ch

    full_overlay[:, y1:y2, x1:x2] = blended_crop

    return full_overlay


# Main visualization

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Load dataset

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

    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"Validation samples: {len(val_dataset)}")


    # Load model

    model = unet.UNetMobileNetV3().to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from: {CHECKPOINT_PATH}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")


    # Take first validation batch

    batch = next(iter(val_loader))

    global_scene_model, full_scene_original, local_crop, target_ball, mask, crop_box = batch

    global_scene_model = global_scene_model.to(device)
    local_crop = local_crop.to(device)
    target_ball = target_ball.to(device)
    mask = mask.to(device)


    # Predict

    with torch.no_grad():
        prediction = model(local_crop, global_scene_model)

    # Expand mask to 3 channels
    mask_3ch = mask.expand(-1, 3, -1, -1)

    # Keep only the ball region
    gt_ball = target_ball * mask_3ch
    pred_ball = prediction * mask_3ch

    # Overlay on the local crop
    gt_overlay = local_crop * (1 - mask_3ch) + gt_ball
    pred_overlay = local_crop * (1 - mask_3ch) + pred_ball


    # Save current visualization, unchanged

    save_image(
        local_crop.cpu(),
        os.path.join(OUTPUT_DIR, "01_local_crop.jpg"),
        nrow=3,
        normalize=True
    )

    save_image(
        gt_ball.cpu(),
        os.path.join(OUTPUT_DIR, "02_ground_truth_ball.jpg"),
        nrow=3,
        normalize=True
    )

    save_image(
        pred_ball.cpu(),
        os.path.join(OUTPUT_DIR, "03_predicted_ball.jpg"),
        nrow=3,
        normalize=True
    )

    save_image(
        gt_overlay.cpu(),
        os.path.join(OUTPUT_DIR, "04_ground_truth_overlay.jpg"),
        nrow=3,
        normalize=True
    )

    save_image(
        pred_overlay.cpu(),
        os.path.join(OUTPUT_DIR, "05_predicted_overlay.jpg"),
        nrow=3,
        normalize=True
    )


    # Save per-sample comparison panels

    num_samples = min(BATCH_SIZE, local_crop.size(0))

    for i in range(num_samples):
        save_path = os.path.join(OUTPUT_DIR, f"sample_{i+1}_comparison.jpg")

        save_sample_panel(
            local_crop=local_crop[i].cpu(),
            gt_ball=gt_ball[i].cpu(),
            pred_ball=pred_ball[i].cpu(),
            gt_overlay=gt_overlay[i].cpu(),
            pred_overlay=pred_overlay[i].cpu(),
            save_path=save_path,
        )


    # Save 6 global scenes with predicted ball inserted

    global_pred_overlays = []

    for i in range(num_samples):
        full_overlay = paste_ball_into_global_scene(
            global_scene=full_scene_original[i].cpu(),
            pred_ball=pred_ball[i].cpu(),
            mask=mask[i].cpu(),
            crop_box=crop_box[i]
        )

        global_pred_overlays.append(full_overlay)

        save_image(
            full_overlay,
            os.path.join(OUTPUT_DIR, f"global_scene_{i+1}_with_predicted_ball.jpg"),
            normalize=True
        )

    global_pred_overlays = torch.stack(global_pred_overlays, dim=0)

    save_image(
        global_pred_overlays,
        os.path.join(OUTPUT_DIR, "06_global_scenes_with_predicted_ball.jpg"),
        nrow=3,
        normalize=True
    )

    print(f"Saved final visualizations in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()