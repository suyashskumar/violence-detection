# Video Violence Classification README

This project is a Python-based video violence detection tool that classifies uploaded videos as either violent or non-violent by analyzing extracted video frames. It uses HOG (Histogram of Oriented Gradients) feature extraction and a Support Vector Machine (SVM) classifier for frame classification.

## Prerequisites

1. Python 3.x
2. The following libraries:
    - OpenCV
    - Scikit-learn
    - Matplotlib
    - imbalanced-learn

### Install Required Libraries
To install the libraries, run:
```python
!pip install opencv-python scikit-learn matplotlib imbalanced-learn
```

## How to Use

### Step-by-Step Guide

1. **Upload Video**  
   The script prompts you to upload an `.mp4` video. Use the file upload feature to load the video into the program.

2. **Extract and Display Frames**  
   The `extract_and_display_frames()` function captures frames from the uploaded video, skipping frames to reduce processing time. These frames are saved locally in the `uploaded_video_frames` directory and displayed.

3. **Extract Features**  
   The `extract_hog_features()` function uses HOG to extract features from each frame for violence classification. Frames are labeled as non-violent by default, with an option to add simulated violent labels if needed.

4. **Handle Class Imbalance**  
   Class imbalance is managed using Random Over-Sampling to balance the dataset if violent frames are present.

5. **Train Random Forest Classifier**  
   A Random Forest classifier is trained on the extracted frame features.

6. **Predict and Evaluate**  
   The model predicts the class of frames in the test set, reporting accuracy and a classification report if both classes are present.

7. **Classify Entire Video**  
   All frames are classified to determine the video's overall class, either violent or non-violent, based on majority vote.

8. **Memory Management**  
   The program uses garbage collection to free up memory.

## Directory Structure

```
project/
├── uploaded_video_frames/    # Directory where extracted frames are stored
└── main_script.py            # The main script for video violence detection
```

## Important Functions

- **`extract_and_display_frames(video_path, output_dir, max_frames, frame_skip)`**  
  Captures frames from the video, resizes them, and saves them in the specified directory.

- **`extract_hog_features(image_path)`**  
  Extracts HOG features from an image file for frame classification.

- **`extract_features_from_all_frames(frame_dir, violent_label)`**  
  Collects features and labels from all frames in a directory for training/testing.

## Example Output

```
Please upload your mp4 video file:
Extracted 60 frames from video.mp4 and saved to uploaded_video_frames
Model Accuracy: 85.00%
The video is classified as: Violent
```

## Notes
- This script is designed for educational purposes and may require modifications for real-world applications.
- Ensure that the video data includes diverse frames to improve model accuracy.
