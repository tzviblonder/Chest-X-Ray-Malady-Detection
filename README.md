# Chest X-Ray Malady Detection
## Overview

This repository contains the implementation of a machine learning model designed to analyze chest X-ray images and detect the presence of various medical conditions. The model is trained on a dataset of labeled X-ray images and leverages deep learning techniques to achieve high accuracy in diagnosing conditions such as pneumonia, lesions, and other lung-related diseases. Where radiologists are unavailable, this model could serve in the backend of an app that takes images of x-rays and outputs accurate predictions.

## Data
The data is the CheXpert dataset, a large public dataset of 224,316 images from 65,240 patients. More information regarding the CheXpert dataset can be found on the Stanford Machine Learning Group's website (https://stanfordmlgroup.github.io/competitions/chexpert/) as well as the CheXpert paper (https://arxiv.org/abs/1901.07031).
<p>The dataset is broken into train and test sets, with 61,313 patients (212,220 images) in the train set and 3,227 patients (11,194) to test. Each image is 512x512x1 pixels. While training, the data is augmented in order to prevent overfitting. </p>

## Model
The model is a convolutional neural network (CNN). It has a series of convolutional blocks, each consisting of a convolutional layer, batch normalization, max pooling, and dropout, which reduce dimensionality while increasing depth. These are followed by fully-connected layers as the classifier. The model used binary crossentropy loss and Adam optimizer, with a learning rate starting at 1e-3 and slowly decreasing until 1e-5. It was trained for 60+ hours on a GPU and the weights were downloaded and uploaded in this notebook.

## Results
As shown in the notebook, the model achieved an accuracy of 96% and an AUC score of .9926 on the training data. On the test data, it attains an accuracy of 84.2% and an AUC of .908.

## Environment
Python: 3.10.5\
NumPy: 1.23.5\
Pandas: 1.5.3\
Matplotlib: 3.7.0\
Seaborn: 0.11.2\
TensorFlow: 2.14.0\
Scikit-Learn: 1.2.2
