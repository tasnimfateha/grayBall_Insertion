# Grey Ball Insertion

## Overview

Grey Ball Insertion is a computer vision and deep learning project that focuses on detecting, segmenting, and learning the placement of a grey reference ball within images.

This project trains a deep learning model to generate a realistic grey ball patch for a real scene. The model uses both local scene context and global scene context, then predicts a 256×256 RGB ball region that can later be overlaid back into the image.

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

### The Final Learning Problem is:
Input:
1. local scene crop around the ball position
2. full aligned scene image for global lighting context

Output:
256×256 RGB predicted grey ball patch

### The Project's Structure:
```text
Grey_Ball_Insertion/
│
├── scenes/
├── scenes_shots/
├── illumination_gt/
├── Masked_Crops/
├── checkpoints/
├── training_outputs/
├── runs/
│
├── matching_results.csv
├── Updated_Ball_Data.csv
├── ball_data_modified.csv
│
├── matching_data.py
├── segmentation.py
├── GreyBallInsertionDataset.py
├── UNet_MobileNet.py
├── Training.py
├── model_results.py
├── utils.py
└── sam2.1_l.pt
```

### Main Input Folders

* scenes/ - contains the clean reference scene images, usually as .NEF files.
* scenes_shots/ - contains scene-shot images where the grey ball is visible.
* illumination_gt/ - contains the original ground-truth CSV files for each scene (image_name, circle_x, circle_y, circle_radius).
* Masked_Crops/ - created by segmentation.py,contains cropped 256×256 masked ball images, used as the training targets.

  ### Main CSV Files
* ball_data_modified.csv - contains approximate ball positions before SAM refinement (prepared manually from original ball position CSV files).
* Updated_Ball_Data.csv - created by segmentation.py, the updated annotation file after SAM-based preprocessing, used for future model training. It stores the refined ball center and radius for each processed image.
* matching_results.csv - created by matching_data.py, stores the 3×3 homography matrix for each scene-shot pair. The homography is used to align the clean reference scene with the corresponding scene-shot image.
  
### Step 1 - Prepare the Dataset

Here we used a script *DataImport.py*, but since the website wasn't warking - so mostly we did it manually. 
Downloads images into:
* scenes/ (reference scene images without drone)
* scenes_shots/ (images with drone + ball)
* illumination_gt/ (original ball position CSV files)
* sam2.1_l.pt (the SAM checkpoint required for segmentation)

### Step 2 - Compute Geometric Alignment

Run: python matching_data.py
It creates: matching_results.csv

The purpose of this step is to align the clean reference scene with the scene-shot image, because even small camera differences can make the images slightly misaligned.

### Step 3 - Image Preprocessing, Segment and Crop the Grey Ball

Run: python segmentation.py, uses Segment Anything Model (SAM).
It creates:
* Masked_Crops/
* Updated_Ball_Data.csv

The script crops and masks images to isolate the ball and saves the output. So this step prepares the training targets. The model does not directly learn from the full shot image. It learns from the cropped masked ball image.

### Step 4 - Create the Training Dataset

Run: python GreyBallInsertionDataset.py. This file checks whether the dataset can be loaded correctly. It reads: Updated_Ball_Data.csv, matching_results.csv, scenes/ and Masked_Crops/. For each samples, it returns four tensors: global_scene, local_crop, target_ball, mask.

The dataset file loads the clean scene, applies the homography, crops around the ball position, loads the masked target ball crop, creates a binary mask, converts everything to PyTorch tensors, and normalizes the global scene for MobileNetV3.

### Step 5 - Define the Model Architecture

Run: python UNet_MobileNet.py.
This checks the model input and output shapes. The relevant neural network model is: U-Net + MobileNetV3

### Step 6 - Train the model

Run: python Training.py. This file trains UNetMobileNetV3.
It reads: Updated_Ball_Data.csv, matching_results.csv, scenes/, Masked_Crops/.
Then it creates a train/validation split: 80% training and 20% validation.

The loss is masked L1 loss. This means the model is punished only for errors inside the ball area, while the black background is ignored. The training script also saves TensorBoard logs, checkpoints, and visual prediction examples.

#### Training outputs
After running Training.py, the following folders are created.

* checkpoints/ - stores the model with the lowest validation loss.
* training_outputs/ - contains visual training examples for training and validation both ground truth and prediction.
These images help check whether the model prediction is becoming similar to the real grey ball crop.

* runs/

Contains TensorBoard logs:

runs/grey_ball_insertion/

To view the loss curves:

tensorboard --logdir runs

### Step 7 - Visualize final predictions
Run: python overlay_model_results.py. 
The final visualization step loads the best trained checkpoint and applies the model to validation samples. For each sample, the predicted grey ball patch is masked and overlaid onto the local scene crop. The saved comparison images allow us to evaluate how well the predicted ball matches the ground-truth ball and the scene illumination.
