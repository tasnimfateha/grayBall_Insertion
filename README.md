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