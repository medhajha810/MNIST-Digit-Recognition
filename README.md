# MNIST Digit Recognition: CNN vs. Random Forest

This project compares the performance of **Convolutional Neural Networks (CNN)** and **Random Forest Classifiers** on the MNIST dataset, focusing on **test accuracy**, **training time**, **classification report**, and **confusion matrix**.

## Project Overview

The MNIST dataset is a well-known collection of 28x28 grayscale images of handwritten digits (0-9). The goal of this project is to evaluate and compare two machine learning models:

1. **Convolutional Neural Network (CNN)** - A deep learning model for image classification tasks.
2. **Random Forest Classifier** - A classical machine learning model that can also be used for classification tasks.

### Models Used
- **CNN**: A deep learning architecture with multiple convolutional and max-pooling layers followed by fully connected layers.
- **Random Forest**: An ensemble method based on decision trees that aggregates the results of multiple trees to make predictions.

## Features

- **Model Comparison**: The models are compared in terms of:
  - Test accuracy
  - Training time
  - Classification report (precision, recall, F1-score)
  - Confusion matrix

- **Visualization**: 
  - Bar charts comparing accuracy and training time.
  - Heatmaps of confusion matrices for each model.

## Requirements

To run this project, you will need the following libraries:
- TensorFlow (for CNN)
- Scikit-learn (for Random Forest)
- Numpy
- Matplotlib
- Seaborn

You can install the dependencies using pip:

```bash
pip install tensorflow scikit-learn numpy matplotlib seaborn
```

# Datasets

The project uses the **MNIST dataset**, which can be loaded directly from TensorFlow/Keras.

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```
The dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0-9).

## Running the Script

### Training and Evaluation:
The script loads the MNIST dataset, preprocesses it, and trains a CNN and a Random Forest model. The models are evaluated based on test accuracy.

### Metrics:
- **Classification Report:** Provides precision, recall, and F1-score for each class (digit).
- **Confusion Matrix:** Displays misclassifications and correctly predicted digits.

### Visualization:
- Bar plots comparing the accuracy and training time of both models.
- Heatmaps of the confusion matrix for both models.

## Results

After running the script, you will get the following results:

- **Test Accuracy:** The percentage of correct predictions on the test set.
- **Classification Report:** Detailed metrics (precision, recall, F1-score) for each class.
- **Confusion Matrix:** A heatmap showing the distribution of correct and incorrect predictions.
- **Training Time:** The time taken to train each model.

### Example Output

```yaml
CNN Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       980
           1       0.99      0.99      0.99      1135
           2       0.98      0.99      0.99      1032
           ...
Random Forest Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.97      0.97       980
           1       0.98      0.97      0.98      1135
           2       0.96      0.97      0.97      1032
           ...
```

#### Confusion Matrix
- **CNN Confusion Matrix:** Visualized with a heatmap showing where the model made correct and incorrect predictions.
- **Random Forest Confusion Matrix:** Similar heatmap for the Random Forest model.
