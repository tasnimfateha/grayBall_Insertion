import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd 
import rawpy  
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class GreyBallInsertionDataset(Dataset):
    """
    Dataset for the Grey Ball Insertion task.

    For each sample, the dataset returns:

    1. global_scene:
       Full aligned reference scene resized to 224x224.
       This is used by MobileNetV3 to extract global lighting/context features.

    2. local_crop:
       256x256 crop from the aligned reference scene around the ball position.
       This is used by U-Net as the local scene context.

    3. target_ball:
       256x256 RGB image of the real masked grey ball.
       This is the target that the network should predict.

    4. mask:
       Binary mask showing where the ball pixels are in the target image.
       This is used for masked loss, so the model focuses only on the ball.
    """

    def __init__(
        self,
        ball_csv_path,
        homography_csv_path,
        scenes_dir,
        masked_targets_dir,
        crop_size=256,
        global_size=224,
        scene_extension=".NEF",
    ):
        self.ball_data = pd.read_csv(ball_csv_path)
        self.homography_data = pd.read_csv(homography_csv_path)

        self.scenes_dir = Path(scenes_dir)
        self.masked_targets_dir = Path(masked_targets_dir)

        self.crop_size = crop_size
        self.global_size = global_size
        self.scene_extension = scene_extension

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.radius_column = self._find_radius_column()

    def __len__(self):
        return len(self.ball_data)

    def __getitem__(self, idx):
        row = self.ball_data.iloc[idx]

        shot_id = Path(str(row["image_name"])).stem
        scene_id = self._find_scene_for_shot(shot_id)

        cx = int(row["circle_x"])
        cy = int(row["circle_y"])

        scene_image = self._load_scene_image(scene_id)
        homography = self._load_homography(scene_id, shot_id)
        aligned_scene = self._apply_homography(scene_image, homography)

        # ----------------------------
        # Local crop
        # ----------------------------
        local_crop = self._crop_around_point(aligned_scene, cx, cy, self.crop_size)

        # Target masked ball
        target_ball = self._load_target_ball(scene_id, shot_id)

        # Mask
        mask = self._create_mask_from_target(target_ball)

        # ----------------------------
        # Global scene
        # ----------------------------
        # Model input: resized
        global_scene_model = cv2.resize(
            aligned_scene,
            (self.global_size, self.global_size),
            interpolation=cv2.INTER_AREA
        )
        global_scene_model = self._to_tensor(global_scene_model)
        global_scene_model = (global_scene_model - self.mean) / self.std

        # Original full-resolution scene for final visualization
        full_scene_original = torch.from_numpy(aligned_scene).permute(2, 0, 1).float() / 255.0

        # ----------------------------
        # Crop coordinates
        # ----------------------------
        half = self.crop_size // 2
        x1 = max(cx - half, 0)
        y1 = max(cy - half, 0)
        x2 = x1 + self.crop_size
        y2 = y1 + self.crop_size
        crop_box = torch.tensor([x1, y1, x2, y2], dtype=torch.long)

        # Convert to tensors
        local_crop = self._to_tensor(local_crop)
        target_ball = self._to_tensor(target_ball)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return global_scene_model, full_scene_original, local_crop, target_ball, mask, crop_box

    def _find_scene_for_shot(self, shot_id):
        """
        Find which scene folder contains this shot.

        Example:
            masked_crops/rugs/DSC_0719.jpg

        returns:
            rugs
        """
        for scene_folder in self.masked_targets_dir.iterdir():
            if not scene_folder.is_dir():
                continue

            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = scene_folder / f"{shot_id}{ext}"
                if candidate.exists():
                    return scene_folder.name

        raise FileNotFoundError(
            f"Could not find shot '{shot_id}' inside {self.masked_targets_dir}"
        )
    def _find_radius_column(self):
        """
        Find the radius column name in the dataset.
        Handles backwards compatibility with datasets that have the misspelling 'circle_radiuos'.
        """
        if "circle_radius" in self.ball_data.columns:
            return "circle_radius"

        if "circle_radiuos" in self.ball_data.columns:
            return "circle_radiuos"

        raise KeyError(
            "Could not find 'circle_radius' column. Expected 'circle_radius' (or legacy 'circle_radiuos')."
        )

    def _load_scene_image(self, scene_id):
        """
        Load the clean reference scene image.
        Usually this is stored as scenes/scene_id.NEF.
        """
        scene_path = self._find_file(
            self.scenes_dir,
            scene_id,
            extensions=[".NEF", ".nef", ".jpg", ".jpeg", ".png"],
        )

        return self._load_image(scene_path)

    def _load_target_ball(self, scene_id, shot_id):
        """
        Load the preprocessed masked ball crop produced by ball_segment_anything.py.

        Expected path:
        scene_shots_masked/scene_id/shot_id.jpg
        """
        target_dir = self.masked_targets_dir / scene_id

        target_path = self._find_file(
            target_dir,
            shot_id,
            extensions=[".jpg", ".jpeg", ".png"],
        )

        target = self._load_image(target_path)

        if target.shape[:2] != (self.crop_size, self.crop_size):
            target = cv2.resize(
                target,
                (self.crop_size, self.crop_size),
                interpolation=cv2.INTER_AREA,
            )

        return target


    def _apply_homography(self, image, homography):
        """
        Warp the reference scene so it aligns with the scene-shot image.
        """
        height, width = image.shape[:2]

        aligned = cv2.warpPerspective(
            image,
            homography,
            (width, height),
        )

        return aligned
   
    def _crop_around_point(self, image, cx, cy, crop_size):
        """
        Crop a fixed-size square around the ball position.
        If the crop goes outside the image, black padding is added.
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


    def _load_homography(self, scene_id, shot_id):
        """
        Load the 3x3 homography matrix for the current scene-shot.
        Works whether Scene ID is stored as DSC_0001 or DSC_0001.NEF.
        """
        scene_id = str(scene_id).strip()
        shot_stem = Path(str(shot_id)).stem.strip()

        scene_rows = self.homography_data[
            self.homography_data["Scene"].astype(str).str.strip().str.lower()
            == scene_id.lower()
        ]

        if scene_rows.empty:
            raise ValueError(f"No homography rows found for scene '{scene_id}'.")

        scene_rows = scene_rows.copy()
        scene_rows["shot_stem"] = scene_rows["Scene ID"].astype(str).apply(
            lambda x: Path(x).stem.strip().lower()
        )

        matching_rows = scene_rows[
            scene_rows["shot_stem"] == shot_stem.lower()
        ]

        if matching_rows.empty:
            raise ValueError(
                f"No homography found for scene '{scene_id}', shot '{shot_stem}'."
            )

        h_values = matching_rows.iloc[0, 2:11].values.astype(float)
        return h_values.reshape(3, 3)

    def _create_mask_from_target(self, target_image):
        """
        The target image has a black background and the grey ball pixels.
        Non-black pixels are treated as the ball mask.
        """
        non_black = np.any(target_image > 0, axis=-1)
        mask = non_black.astype(np.float32)

        return mask

    def _to_tensor(self, image):
        """
        Convert RGB image from H x W x C to C x H x W and normalize to [0, 1].
        """
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    def _load_image(self, path):
        """
        Load NEF, JPG, JPEG, or PNG image as RGB.
        Rotate portrait images to landscape to match preprocessing.
        """
        path = Path(path)

        if path.suffix.lower() == ".nef":
            with rawpy.imread(str(path)) as raw:
                image_rgb = raw.postprocess()
        else:
            image_bgr = cv2.imread(str(path))

            if image_bgr is None:
                raise FileNotFoundError(f"Could not load image: {path}")

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Important: match rotation used in matching_data.py and segmentation.py
        height, width = image_rgb.shape[:2]

        if width < height:
            image_rgb = np.rot90(image_rgb).copy()

        return image_rgb

    def _find_file(self, folder, stem, extensions):
        """
        Find a file with a given name and possible extension.
        """
        folder = Path(folder)

        for ext in extensions:
            candidate = folder / f"{stem}{ext}"
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not find file '{stem}' in folder '{folder}' with extensions {extensions}"
        )


if __name__ == "__main__":
    dataset = GreyBallInsertionDataset(
        ball_csv_path="Updated_Ball_Data.csv",
        homography_csv_path="matching_results.csv",
        scenes_dir="scenes",
        masked_targets_dir="Masked_crops",
        crop_size=256,
        global_size=224,
    )

    print("Number of samples:", len(dataset))

    global_scene, local_crop, target_ball, mask = dataset[0]

    print("Global scene shape:", global_scene.shape)
    print("Local crop shape:", local_crop.shape)
    print("Target ball shape:", target_ball.shape)
    print("Mask shape:", mask.shape)

    print("Global scene range:", global_scene.min().item(), global_scene.max().item())
    print("Local crop range:", local_crop.min().item(), local_crop.max().item())
    print("Target ball range:", target_ball.min().item(), target_ball.max().item())
    print("Mask values:", torch.unique(mask))

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(dataloader))
    global_scene_batch, local_crop_batch, target_ball_batch, mask_batch = batch

    print("Batch global scene:", global_scene_batch.shape)
    print("Batch local crop:", local_crop_batch.shape)
    print("Batch target ball:", target_ball_batch.shape)
    print("Batch mask:", mask_batch.shape)

    