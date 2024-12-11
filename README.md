CNN for Image Classification: Dog vs. Cat
This project is a Convolutional Neural Network (CNN) for classifying images as either a Dog or Cat. The model is built using TensorFlow and Keras.

Project Overview
Dataset: The dataset contains images of dogs and cats. The model is trained on 4000 images of dogs and 4000 images of cats for the training set, and evaluated on 1000 images of dogs and 1000 images of cats in the test set.
Goal: The goal of this project is to train a CNN model that can accurately predict whether a given image is of a dog or a cat.

Files Included
cnn_model.py - The main script for building, training, and evaluating the CNN model.
single_prediction.py - A script for making predictions on a single image.
dataset/ - Folder containing the training and test datasets.
training_set/ - Folder containing training images (dogs and cats).
test_set/ - Folder containing test images (dogs and cats).
single_prediction/ - Folder containing images for making predictions.
requirements.txt - List of required Python libraries to run this project.


Running the Code
Training the Model:

Open the cnn_model.py file.
The model will be trained on the provided training_set and tested on the test_set.
The model's performance will be printed after training.
Making a Prediction:

Put the image you want to classify in the single_prediction folder.

Run the single_prediction.py script: python single_prediction.py 


Model Architecture:
First Convolution Layer: Extracts basic features like edges and textures.
Max-Pooling Layer: Reduces the spatial dimensions of the image.
Second Convolution Layer: Captures more complex features.
Fully Connected Layer: Flattens the features for classification.
Output Layer: Predicts the class (Dog or Cat).

Results:
The model is evaluated on the test set to measure its accuracy in classifying images of dogs and cats.
The prediction result is based on the model’s learned features during training.



Requirements:
Python 3.x
TensorFlow 2.x
Keras
NumPy
Matplotlib
