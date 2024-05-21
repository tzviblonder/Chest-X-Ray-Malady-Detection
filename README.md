# Chest X-Ray Malady Detection
## Overview

This repository contains the implementation of a machine learning model designed to analyze chest X-ray images and detect the presence of various medical conditions. The model is trained on a dataset of labeled X-ray images and leverages deep learning techniques to achieve high accuracy in diagnosing conditions such as pneumonia, lesions, and other lung-related diseases. Where radiologists are unavailable, this model could serve in the backend of an app that takes images of x-rays and outputs accurate predictions.

## Data
The data is the CheXpert dataset, a large public dataset of 224,316 images from 65,240 patients. More information regarding the CheXpert dataset can be found on the Stanford Machine Learning Group's website (https://stanfordmlgroup.github.io/competitions/chexpert/) as well as the CheXpert paper (https://arxiv.org/abs/1901.07031).
The dataset is broken into train and test sets, with 61,313 patients (212,220 images) in the train set and 3,227 patients (11,194) to test. 

## Features
    Image Preprocessing: Functions to preprocess chest X-ray images to ensure optimal input for the model.
    Model Architecture: Implementation of a convolutional neural network (CNN) designed for image classification tasks.
    Training Pipeline: Scripts to train the model on a dataset of chest X-ray images, including data augmentation and to address overfitting.
    Evaluation Metrics: Tools to evaluate the model's performance using metrics such as accuracy and AUC (area under the ROC curve).

## Results
As shown in the notebook, the model achieves an accuracy of 96% and an AUC score of .9926 on the training data. On the test data, it attains an accuracy of 84.2% and an AUC of .908.
