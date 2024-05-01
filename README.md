# Toxic Comment Classification with Deep Learning

## Overview

This repository contains a deep learning model developed to classify toxic comments into multiple categories using TensorFlow and Keras. The model employs a Bidirectional LSTM neural network architecture to capture contextual information from comments, resulting in improved classification accuracy.

## Methodology

1. **Data Preprocessing**: Utilized the TextVectorization layer to preprocess textual data, converting comments into integer sequences for model input.

2. **Model Development**:
   - Implemented a Bidirectional LSTM neural network architecture to capture bidirectional context information from comments.
   - Fine-tuned hyperparameters including learning rate, batch size, and dropout rate to optimize model performance.

3. **Model Training and Evaluation**:
   - Trained the model on the training dataset for 2 epochs, leveraging Google Colab for computational resources.
   - Evaluated model performance using metrics such as precision, recall, and F1-score on the test dataset.

## Project Structure

- `data/`: Contains the dataset used for training and evaluation.
- `notebooks/`: Jupyter notebooks detailing the data preprocessing, model development, and evaluation.
- `models/`: Saved model files.
- `README.md`: Overview of the project.

## Requirements

- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn


