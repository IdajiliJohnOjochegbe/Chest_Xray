# Chest X-Ray Classification

This project demonstrates how to use a pre-trained model from Teachable Machine to classify chest X-ray images. The model is capable of identifying different classes of chest X-rays, such as normal and pneumonia. The implementation is done using Keras in Python.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)
- [Code Explanation](#code-explanation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Chest X-ray classification is a critical task in the medical field, aiding in the diagnosis of various conditions such as pneumonia. This project leverages a model trained using Google's Teachable Machine to classify chest X-ray images. The model is deployed and run using Keras, a deep learning framework.

## Dataset

The dataset used to train the model includes chest X-ray images categorized into different classes, such as normal and pneumonia. The images are preprocessed and resized to 224x224 pixels to match the input size expected by the model.

## Model

The model used in this project is a convolutional neural network (CNN) trained using Teachable Machine. The model file (`keras_model.h5`) and the labels file (`labels.txt`) are used to make predictions on new chest X-ray images.

## Code Explanation

The following code demonstrates how to load the model, preprocess input images, and make predictions:

```python
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("/content/drive/MyDrive/model_chest_xray/keras_model.h5", compile=False)

# Load the labels
class_names = open("/content/drive/MyDrive/model_chest_xray/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("/content/drive/MyDrive/Normal/IM-0039-0001.jpeg").convert("RGB")

# Resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
```
## Results

The model provides predictions on chest X-ray images with a confidence score. The results include the predicted class and the associated confidence score.

## Usage

To use this project, follow these steps:

### Clone the repository:

git clone https://github.com/yourusername/chest-xray-classification.git

### Navigate to the project directory:

cd chest-xray-classification

### Install the required packages:

pip install -r requirements.txt

Place your model file (keras_model.h5) and labels file (labels.txt) in the appropriate directory.

Replace the image path in the code with the path to your image.

Run the script:

python classify_xray.py


