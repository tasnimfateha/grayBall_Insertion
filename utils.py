import os
from pathlib import Path

import cv2
import numpy as np
import rawpy
from PIL import Image


IMAGE_EXTENSIONS = [".NEF", ".nef", ".jpg", ".jpeg", ".png"]


def load_image(path, rotate_to_landscape=True):
    """
    Load an image as RGB.

    Supports:
    - NEF raw images
    - JPG / JPEG / PNG images

    Returns:
        image_rgb: NumPy array with shape [H, W, 3]
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() == ".nef":
        with rawpy.imread(str(path)) as raw:
            image_rgb = raw.postprocess()
    else:
        image_rgb = np.array(Image.open(path).convert("RGB"))

    if rotate_to_landscape:
        h, w = image_rgb.shape[:2]
        if w < h:
            image_rgb = np.rot90(image_rgb).copy()

    return image_rgb


def save_rgb_image(path, image_rgb):
    """
    Save an RGB image using OpenCV.

    OpenCV saves images in BGR format, so we convert RGB -> BGR before saving.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def crop_image(image, cx, cy, crop_size=256):
    """
    Crop a square region around point (cx, cy).

    If the crop goes outside image boundaries, black padding is added.

    Args:
        image: RGB image as NumPy array [H, W, 3]
        cx: center x coordinate
        cy: center y coordinate
        crop_size: output crop size

    Returns:
        crop: RGB crop [crop_size, crop_size, 3]
    """

    h, w = image.shape[:2]
    half = crop_size // 2

    x1 = int(cx - half)
    y1 = int(cy - half)
    x2 = int(cx + half)
    y2 = int(cy + half)

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    crop = image[y1:y2, x1:x2]

    if crop.shape[:2] != (crop_size, crop_size):
        crop = cv2.resize(
            crop,
            (crop_size, crop_size),
            interpolation=cv2.INTER_AREA,
        )

    return crop


def apply_homography_transformation(image, homography):
    """
    Apply homography transformation to an image.

    This aligns the clean reference scene with the scene-shot image.
    """

    if homography is None:
        return image

    h, w = image.shape[:2]

    try:
        transformed = cv2.warpPerspective(
            image,
            homography,
            (w, h),
        )
        return transformed

    except cv2.error as error:
        print(f"OpenCV homography error: {error}")
        return image


def create_non_black_mask(image):
    """
    Create a binary mask from a cropped masked ball image.

    Pixels that are not black are treated as ball pixels.

    Returns:
        mask: NumPy array [H, W], values 0 or 1
    """

    mask = np.any(image > 0, axis=-1).astype(np.uint8)
    return mask


def find_file(folder, stem, extensions=None):
    """
    Find a file by name without knowing its extension.

    Example:
        find_file("scenes", "seat_rows")
        can find:
        scenes/seat_rows.NEF
        scenes/seat_rows.jpg
        scenes/seat_rows.png
    """

    if extensions is None:
        extensions = IMAGE_EXTENSIONS

    folder = Path(folder)

    for ext in extensions:
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find file '{stem}' in {folder} with extensions {extensions}"
    )


def get_scene_paths(scenes_dir="scenes"):
    """
    Collect clean reference scene image paths.

    Returns:
        dictionary:
        {
            scene_id: path_to_scene_image
        }
    """

    scenes_dir = Path(scenes_dir)
    scene_paths = {}

    for file in scenes_dir.iterdir():
        if file.suffix.lower() in [ext.lower() for ext in IMAGE_EXTENSIONS]:
            scene_id = file.stem
            scene_paths[scene_id] = str(file)

    return scene_paths


def get_masked_crop_paths(masked_dir="masked_crops"):
    """
    Collect masked ball crop image paths.

    Expected structure:
        masked_crops/
            scene_id/
                shot_id.jpg

    Returns:
        list of tuples:
        (scene_id, shot_id, image_path)
    """

    masked_dir = Path(masked_dir)
    paths = []

    if not masked_dir.exists():
        raise FileNotFoundError(f"Masked crops folder not found: {masked_dir}")

    for scene_folder in masked_dir.iterdir():
        if not scene_folder.is_dir():
            continue

        scene_id = scene_folder.name

        for file in scene_folder.iterdir():
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                shot_id = file.stem
                paths.append((scene_id, shot_id, str(file)))

    return paths


def get_original_shot_paths(shots_dir="scenes_shots"):
    """
    Collect original scene-shot image paths.

    Expected structure:
        scenes_shots/
            scene_id/
                shot_id.NEF

    This is useful for segmentation.py, before masked crops are created.

    Returns:
        list of tuples:
        (scene_id, shot_id, image_path)
    """

    shots_dir = Path(shots_dir)
    paths = []

    if not shots_dir.exists():
        raise FileNotFoundError(f"Scene shots folder not found: {shots_dir}")

    for scene_folder in shots_dir.iterdir():
        if not scene_folder.is_dir():
            continue

        scene_id = scene_folder.name

        for file in scene_folder.iterdir():
            if file.suffix.lower() in [ext.lower() for ext in IMAGE_EXTENSIONS]:
                shot_id = file.stem
                paths.append((scene_id, shot_id, str(file)))

    return paths


def get_radius_column(dataframe):
    """
    Handle both possible radius column names.

    Correct spelling:
        circle_radius
    
    Legacy misspelling (for backwards compatibility):
        circle_radiuos
    """

    if "circle_radius" in dataframe.columns:
        return "circle_radius"

    if "circle_radiuos" in dataframe.columns:
        return "circle_radiuos"

    raise KeyError(
        "No radius column found. Expected 'circle_radius' (or legacy misspelled 'circle_radiuos')."
    )
