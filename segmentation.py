import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import rawpy
from ultralytics import SAM

# Configuration constants
CHECKPOINT_PATH = "sam2.1_l.pt"
ILLUMINATION_GT_DIR = "illumination_gt"
SCENE_SHOTS_DIR = "scenes_shots"
OUTPUT_DIR = "Masked_Crops"
OUTPUT_CSV = "Updated_Ball_Data.csv"
ROI_RADIUS = 300
CROP_SIZE = 256
DEBUG = False
DEBUG_DIR = "debug_sam"


def build_shot_index(scene_shots_dir: str):
    """
    Build a mapping of scene ID to shot file paths.
    """
    shot_index = {}
    for scene_dir in sorted(Path(scene_shots_dir).iterdir()):
        if scene_dir.is_dir():
            scene_id = scene_dir.name
            shot_index[scene_id] = {
                file_path.stem: str(file_path) for file_path in scene_dir.iterdir() if file_path.is_file()
            }
    return shot_index


def load_image_with_rotation(path: str):
    """
    Load an image and rotate it if necessary.
    """
    ext = Path(path).suffix.lower()
    bgr = cv2.imread(path) if ext != ".nef" else rawpy.imread(path).postprocess()
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    if bgr.shape[1] < bgr.shape[0]:
        bgr = np.rot90(bgr).copy()
    return bgr


def crop_image(image: np.ndarray, cx: int, cy: int, crop_size: int = 256, return_coords: bool = False):
    h, w = image.shape[:2]
    half = crop_size // 2

    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w)
    y2 = min(cy + half, h)

    crop = image[y1:y2, x1:x2]

    if return_coords:
        return crop, x1, y1

    return crop


def erode_keep_ball(mask: np.ndarray, ball_radius: float) -> np.ndarray:
    """
    Erode and dilate the mask to refine the detected ball's boundary.
    """
    kernel_size = max(1, int(0.5 * ball_radius) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask, kernel, iterations=1)
    return cv2.dilate(eroded, kernel, iterations=1)


def fit_circle(mask: np.ndarray) -> np.ndarray:
    """
    Fit a circle to the detected mask area.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask, dtype=np.uint8)
    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    circle_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.circle(circle_mask, (int(x), int(y)), int(radius), 1, -1)
    return circle_mask

def predict_mask(image: np.ndarray, model, point: list, fallback_radius: int = 50):
    """
    Predict a mask for the grey ball using the SAM model.
    """
    roi, x1, y1 = crop_image(
        image,
        point[0],
        point[1],
        ROI_RADIUS * 2,
        return_coords=True
    )

    prompt_point = [int(point[0] - x1), int(point[1] - y1)]

    results = model(roi, points=[prompt_point], labels=[1])

    if results and results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return make_fallback_mask(image.shape, point[0], point[1], fallback_radius)

        cnt = max(contours, key=cv2.contourArea)
        (_, _), radius = cv2.minEnclosingCircle(cnt)

        cleaned_mask = erode_keep_ball(mask, radius)
        circle_mask = fit_circle(cleaned_mask)

        return extend_mask(circle_mask, image.shape[0], image.shape[1], x1, y1)

    return make_fallback_mask(image.shape, point[0], point[1], fallback_radius)


def extend_mask(mask: np.ndarray, full_h: int, full_w: int, x1: int, y1: int) -> np.ndarray:
    """
    Place a cropped mask back into the original image size.
    """
    full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
    mh, mw = mask.shape
    full_mask[y1:y1 + mh, x1:x1 + mw] = mask
    return full_mask


def make_fallback_mask(shape, cx: int, cy: int, radius: int) -> np.ndarray:
    """
    Create a fallback circular mask if the model fails.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), max(8, radius), 1, -1)
    return mask

def calculate_mask_params(mask: np.ndarray):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask, (0, 0), 0

    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)

    return mask, (int(round(x)), int(round(y))), int(round(radius))

def process_scene_csv(csv_path: str, shot_index: dict, model) -> list:
    scene_id = Path(csv_path).stem
    scene_rows = []

    if scene_id not in shot_index:
        print(f"Skipping scene {scene_id}, no folder in {SCENE_SHOTS_DIR}")
        return scene_rows

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Only keep the necessary columns (image_name, circle_x, circle_y, circle_radiuos)
    required_cols = ['image_name', 'circle_x', 'circle_y', 'circle_radiuos']
    df = df[required_cols]

    # Ensure image_name is in the correct format (removing extensions)
    df["image_name"] = df["image_name"].astype(str).apply(lambda x: os.path.splitext(x)[0])

    scene_output_dir = os.path.join(OUTPUT_DIR, scene_id)
    os.makedirs(scene_output_dir, exist_ok=True)

    # Loop through each row in the dataframe
    for idx, row in df.iterrows():
        # Get the necessary values from the row
        shot_id = row["image_name"]
        shot_path = shot_index[scene_id].get(shot_id)

        if shot_path is None:
            print(f"[{scene_id}] missing shot file for {shot_id}")
            continue

        try:
            cx = int(round(float(row["circle_x"])))
            cy = int(round(float(row["circle_y"])))
            radius0 = int(round(float(row["circle_radiuos"])))
        except Exception as e:
            print(f"[{scene_id}] invalid coordinates in row for {shot_id}: {e}")
            continue

        # Load the image and predict the mask
        try:
            image = load_image_with_rotation(shot_path)
            mask = predict_mask(image, model, [cx, cy], radius0)

            # Handle the mask and calculate center and radius
            _, center, radius = calculate_mask_params(mask) if mask is not None else (None, (cx, cy), radius0)

            # Crop the masked image
            masked_image = image * mask[:, :, np.newaxis]
            cropped_masked = crop_image(masked_image, center[0], center[1], CROP_SIZE)

            # Save the cropped image
            output_path = os.path.join(scene_output_dir, f"{shot_id}.jpg")
            cv2.imwrite(output_path, cropped_masked)

            # Append the processed row data
            scene_rows.append({
                "image_name": shot_id,
                "circle_x": int(center[0]),
                "circle_y": int(center[1]),
                "circle_radiuos": int(radius)
            })

            print(f"[{scene_id}] {idx + 1}/{len(df)} {shot_id} -> center=({center[0]}, {center[1]}), radius={radius}")

        except Exception as e:
            print(f"[{scene_id}] failed on {shot_id}: {e}")
    return scene_rows


def main():
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Missing SAM checkpoint: {CHECKPOINT_PATH}")
    if not os.path.isdir(SCENE_SHOTS_DIR):
        raise FileNotFoundError(f"Missing folder: {SCENE_SHOTS_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = SAM(CHECKPOINT_PATH)
    shot_index = build_shot_index(SCENE_SHOTS_DIR)
    
    all_rows = []
    for csv_path in Path(ILLUMINATION_GT_DIR).glob("*.csv"):
        print(f"Processing {csv_path.stem}")
        rows = process_scene_csv(csv_path, shot_index, model)
        all_rows.extend(rows)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False)
        print(f"Processed and saved data to {OUTPUT_CSV}")
    else:
        print("No valid rows processed.")


if __name__ == "__main__":
    main()