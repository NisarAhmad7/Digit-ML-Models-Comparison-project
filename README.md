
# Digit ML Models Comparison project

## Project Overview

This project compares multiple Machine Learning models for the task of handwritten digit classification using the classic Digits dataset from scikit-learn.

The main objective is to analyze how different models perform when trained on the same data pipeline, including feature scaling and dimensionality reduction (PCA).

## Problem Statement

Handwritten digit recognition is a multi-class classification problem where each image represents a digit from 0 to 9.

This task is a fundamental benchmark problem in machine learning and computer vision.

## Dataset

Source: sklearn.datasets.load_digits

Samples: 1,797

Features: 64 pixel intensity values (8×8 images)

Classes: 10 (digits 0–9)

The dataset is widely used for evaluating classical ML algorithms on image-like numerical data.

## Data Preprocessing

To ensure optimal performance and fair comparison, the following preprocessing steps are applied:

## Train/Test Split

70% Training

30% Testing

train_test_split(test_size=0.3)

## Feature Scaling

MinMaxScaler scales pixel values to the range [0, 1]

Especially important for distance-based and neural models

MinMaxScaler()

## Dimensionality Reduction (PCA)

Principal Component Analysis (PCA)

Number of components: 32

Reduces dimensionality while preserving essential information

PCA(n_components=32)


This step improves training efficiency and reduces noise.

## Models Implemented

The following machine learning models are trained and evaluated:

Model	Description
Random Forest (RF)	Ensemble of decision trees
Support Vector Machine (SVM)	Linear kernel classifier
Artificial Neural Network (ANN)	Multi-layer perceptron

Each model is trained using reasonable hyperparameters without heavy tuning to maintain a fair comparison.

## Evaluation Metrics

Performance is evaluated using standard multi-class classification metrics:

Accuracy – Overall classification correctness

Precision (Weighted) – Reliability of predictions across all classes

Recall (Weighted) – Ability to correctly identify all classes

A shared evaluation function is used to compute metrics consistently across models.

## Visualization

Two bar charts are generated:

- Training Accuracy Comparison

Shows how well each model fits the training data.

- Testing Accuracy Comparison

Demonstrates the model’s generalization capability.

These visualizations help identify:

Overfitting

Model robustness

Relative performance differences

## Experimental Design Philosophy

Same dataset

Same preprocessing (scaling + PCA)

Same evaluation metrics

Only the model architecture changes

This ensures a controlled and fair comparison.

## Key Observations

PCA significantly reduces dimensionality without major performance loss

SVM performs well on linearly separable representations

Random Forest offers strong baseline performance

Neural networks benefit from scaled and reduced features

Classical ML models remain competitive on structured image data

## What I Learned

Importance of dimensionality reduction in image-based ML tasks

How different models respond to PCA-transformed data

Practical comparison of ensemble, margin-based, and neural models

Evaluation of multi-class classification using weighted metrics

## Project Structure
Digit-ML-Models-Comparison/
│
├── main.py
├── README.md
└── requirements.txt

## Technologies Used

Python

Scikit-learn

Matplotlib

NumPy

## Future Improvements

Add confusion matrix visualization

Perform hyperparameter tuning (GridSearchCV)

Try non-linear SVM kernels

Compare results with CNN-based approaches

Add cross-validation

## Author

Nisar Ahmad Zamani
Machine Learning Engineer 
