"""
Create matching_results.csv.

The CSV stores the 3x3 homography matrix between each scene image
and each matching scene shot image.
"""

import csv
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import rawpy


# -----------------------------
# Paths
# -----------------------------

SCENES_DIR = Path("scenes")
SCENE_SHOTS_DIR = Path("scenes_shots")

OUTPUT_CSV = Path("matching_results.csv")


# -----------------------------
# Settings
# -----------------------------

SCENE_NAMES = [
    "A&W1", "A&W2", "A&W3",
    "Harbour1", "Harbour2", "Harbour3", "Harbour4",
    "SFU_art", "blue_ceiling", "dining_area", "downtown_smith",
    "edu_area", "foodcourt_mcnz", "hallway", "image_theater",
    "owl_statue", "playground", "rugs", "seat_rows", "study_area",
    "stump", "subway1", "subway2", "theater", "tree_tunel",
    "uncle_fatih1", "uncle_fatih2", "under_tree2", "wall_art",
    "wall_hallway", "wall_lab"
]

N_FEATURES = 1000
MIN_MATCHES = 8
RANSAC_THRESHOLD = 5.0


# -----------------------------
# Image loading
# -----------------------------

def load_raw_as_gray(path):
    """
    Load a RAW image, convert it to grayscale, and rotate it if needed.
    """

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess()

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape

    if width < height:
        gray = np.rot90(gray)

    return np.ascontiguousarray(gray)


def load_scene(scene_name):
    """
    Load the main scene image.
    """

    scene_path = SCENES_DIR / f"{scene_name}.NEF"
    return load_raw_as_gray(scene_path)


def load_scene_shot(scene_name, shot_filename):
    """
    Load one scene shot image.
    """

    shot_path = SCENE_SHOTS_DIR / scene_name / shot_filename
    return load_raw_as_gray(shot_path)


# -----------------------------
# Homography calculation
# -----------------------------

def compute_homography(scene_img, shot_img):
    """
    Find the homography matrix from the scene image to the scene shot image.
    """

    orb = cv2.ORB_create(nfeatures=N_FEATURES)

    kp1, des1 = orb.detectAndCompute(scene_img, None)
    kp2, des2 = orb.detectAndCompute(shot_img, None)

    if des1 is None or des2 is None:
        raise ValueError("No descriptors found in one of the images.")

    if len(kp1) < 4 or len(kp2) < 4:
        raise ValueError("Not enough keypoints found.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    if len(matches) < MIN_MATCHES:
        raise ValueError(f"Not enough matches found: {len(matches)}")

    matches = sorted(matches, key=lambda match: match.distance)

    pts_scene = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_shot = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(
        pts_scene,
        pts_shot,
        cv2.RANSAC,
        RANSAC_THRESHOLD
    )

    if H is None:
        raise ValueError(" Could not be matched.")

    return H


# -----------------------------
# Processing
# -----------------------------

def process_one_shot(scene_name, scene_img, shot_filename):
    """
    Process one scene shot and return one CSV row.
    """

    shot_img = load_scene_shot(scene_name, shot_filename)
    H = compute_homography(scene_img, shot_img)

    flattened_H = H.flatten().tolist()

    return [scene_name, shot_filename] + flattened_H


def get_shot_files(scene_name):
    """
    Get all files inside scenes_shots/scene_name.
    """

    folder_path = SCENE_SHOTS_DIR / scene_name

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    return sorted([
        file.name
        for file in folder_path.iterdir()
        if file.is_file()
    ])


def save_csv(rows):
    """
    Save successful matching results.
    """

    header = ["Scene", "Scene ID"] + [
        f"H_{i}{j}"
        for i in range(3)
        for j in range(3)
    ]

    with open(OUTPUT_CSV, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    transformation_rows = []

    max_workers = min(8, os.cpu_count() or 1)

    for scene_name in SCENE_NAMES:
        print(f"\nLoading scene: {scene_name}")

        try:
            scene_img = load_scene(scene_name)
            shot_files = get_shot_files(scene_name)

        except Exception as error:
            print(f"Failed to load scene {scene_name}: {error}")
            continue

        print(f"Found {len(shot_files)} shots for {scene_name}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_one_shot,
                    scene_name,
                    scene_img,
                    shot_filename
                ): shot_filename
                for shot_filename in shot_files
            }

            for future in as_completed(futures):
                shot_filename = futures[future]

                try:
                    row = future.result()
                    transformation_rows.append(row)
                    print(f"Done: {scene_name}/{shot_filename}")

                except Exception as error:
                    print(f"Failed: {scene_name}/{shot_filename}, {error}")
                    continue

    transformation_rows.sort(key=lambda row: (row[0], row[1]))

    save_csv(transformation_rows)

    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()