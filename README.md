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

### Step 1 - Download dataset
Here we used a script *DataImport.py* + we attach a link to a Google Drive (in case the website is not working). 
Downloads images into:

scenes/ (reference scene images without drone)
scenes_shots/ (images with drone + ball)

### Step 2 - Detect and crop the ball
Script:

ball_segment_anything.py

Uses Segment Anything Model (SAM).

Tasks:

Detect the ball
Segment it
Crop the image around it
Save coordinates

Output:

ball_data_modified.csv

which stores:

x_center
y_center
radius

for every image.

From the README:
the script crops and masks images to isolate the ball and saves center + radius
