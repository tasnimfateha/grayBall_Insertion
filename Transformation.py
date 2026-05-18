"""
Example usage of how to use the homographic matrix obtained by "Feature_Matching_fast.py"
(include in Dataset Creation Code)
"""

import cv2
import numpy as np
import rawpy
import os
import csv


def load_scene(scene):
    """
    Load the scene image for processing it
    input: name of scene
    output: image of scene (grey scaled, rotated to have same orientation)
    """
    scene_path = os.path.join("scenes", scene + ".NEF")
    with rawpy.imread(scene_path) as raw:
        rgb = raw.postprocess()
    scene_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = scene_img.shape
    if width < height:
        scene_img = np.rot90(scene_img)
    return scene_img


def load_scene_shot(scene, scene_id):
    """
    Load the scene shot image
    input: name of scene and scene ID
    output: loaded scene shot image
    """
    scene_shot_path = os.path.join("scenes_shots", scene, scene_id)
    with rawpy.imread(scene_shot_path) as raw:
        rgb = raw.postprocess()
    scene_shot_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width= scene_shot_img.shape
    if width < height:
        scene_shot_img = np.rot90(scene_shot_img)
    return scene_shot_img


def apply_homography_transformation(scene_img, H):
    """
    apply the homography transformation to the scene
    inupt: scene that needs to be transformed, homographic matrix
    ouput: transformed image
    """
    try:
        # Get the dimensions of the scene shot image
        rows, cols = scene_img.shape

        # Use the homography matrix to transform the scene image
        transformed_img = cv2.warpPerspective(scene_img, H, (cols, rows))

        return transformed_img
    except:
        return None

def read_homography_from_csv(file_path, scene, scene_id):
    """
    Read homography matrix from the CSV file
    input: path of csv-file, name of scene, scene ID
    ouput: Homographic matrix
    """
    
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
           
            # Check if the row matches the desired scene and scene_id
            if row[0] == scene and row[1] == scene_id:
               
                # Extract the homography values (skip the first two columns: Scene, Scene ID)
                H_values = list(map(float, row[2:]))
                
                # Reshape the flattened values into a 3x3 matrix
                H = np.array(H_values).reshape(3, 3)
                return H
    return None  

def main():
    folder = 'A&W1'  # Example folder
    scene_img = load_scene(folder)

    scene_id = 'DSC_0547.NEF'  # Example scene shot file
    scene_shot_img = load_scene_shot(folder, scene_id)

    # Path to the CSV file containing homography matrices
    csv_file_path = 'homography_transformation.csv'

    # Read the homography matrix for the given scene and scene shot
    H = read_homography_from_csv(csv_file_path, folder, scene_id)

    if H is not None:
        # Apply the homography to transform the scene shot image
        transformed_img = apply_homography_transformation(scene_img, H)

        # Resize images to fit the window
        scene_shot_img = cv2.resize(scene_shot_img, (600, 400))
        transformed_img = cv2.resize(transformed_img, (600, 400))
        scene_img = cv2.resize(scene_img, (600, 400))

        # Show the images for comparison
        scene_shot_img = np.rot90(scene_shot_img)
        cv2.imshow('Scene Shot', scene_shot_img)
        cv2.waitKey(0)

        cv2.imshow('Transformed Scene', transformed_img)
        cv2.waitKey(0)

        cv2.imshow('Scene', scene_img)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
    else:
        print(f"No homography matrix found for {folder}/{scene_id}")

#Execute program
if __name__ == '__main__':
    main()
