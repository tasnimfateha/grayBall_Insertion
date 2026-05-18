# Grey Ball Insertion

## Overview

Grey Ball Insertion is a computer vision and deep learning project that focuses on detecting, segmenting, and learning the placement of a grey reference ball within images.

The pipeline combines data preprocessing, segmentation using Segment Anything (SAM), and training a neural network model (U-Net with a MobileNet backbone) to understand spatial and visual characteristics of the ball in different scenes.

This project is useful for tasks such as illumination estimation, scene understanding, and object-aware image processing.

## How to Run

First, create a virtual environment to avoid dependency problems.

### For Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### For Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
```
Then Install all required Python packages using:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## The Project's Pipeline

### Step 1 - Prepare the Dataset

Here we used a script *DataImport.py* + we attach a link to a Google Drive (in case the website is not working), so mostly we did it manually. 
Downloads images into:
scenes/ (reference scene images without drone)
scenes_shots/ (images with drone + ball)

### Step 2 - Preprocess the images

Script: *ball_segment_anything.py*, uses Segment Anything Model (SAM).
The script crops and masks images to isolate the ball and saves the output: x_center, y_center, radius in *ball_data_modified.csv* for every image.

### Step 3 - Compute geometric alignment

Script: *Feature_Matching_fast.py* to align scene image and scene shot. Since camera is fixed and drone moves we need to compute homography transformation, which is saved in: *homography_transformations.csv*.

### Step 4 - Create the training dataset

Script: *DatasetCreation.py* builds a PyTorch Dataset class.
For each sample the input is a scene image and the target is a mask of grey ball.
The mask is created using the stored center and radius from the CSV.

### Step 5 - Helper functions

Script: *Utils.py* as a support file - it support image loading, path handling and transformations or mask utilities used by other files.

### Step 6 - Define the model

Script: *unet.py* , contains the neural network architecture.
The relevant model is: U-Net + MobileNetV3

### Step 7 - Train the model

Script: *train_unet_mobilenet.py* uses encoder architecture MobileNetV3 and decoder architecture U-Net.

Input image ⮕ MobileNetV3 (feature extraction) ⮕ U-Net decoder ⮕ segmentation mask

Loss function likely: BCE loss or Dice loss (will have to correct that!). Goal is to minimize the difference between predicted mask and ground truth mask.

### Step 8 - Visualize predictions

Script: *overlay_model_results.py*, overlays predicted mask + original image to see whether the model detected the ball correctly (location, ball's shape and etc).
