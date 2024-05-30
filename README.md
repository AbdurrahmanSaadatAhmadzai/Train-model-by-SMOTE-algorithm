# Train-model-by-SMOTE-algorithm
This repository contains a Python script demonstrating the application of Synthetic Minority Over-sampling Technique (SMOTE) to handle imbalanced datasets in machine learning. Imbalanced datasets are common in various real-world scenarios, where one class significantly outnumbers the other(s), leading to biased models that may perform poorly on minority classes.

The script uses the following steps:

Loads an imbalanced dataset from a CSV file.
Visualizes the class distribution of the imbalanced dataset.
Splits the dataset into features and target variables.
Divides the data into training and testing sets.
Applies SMOTE (Synthetic Minority Over-sampling Technique) to the training data to balance the classes.
Trains a Random Forest classifier using the balanced data after SMOTE.
Evaluates the model's performance on the imbalanced test data using accuracy and classification report.
This script serves as a practical guide for addressing class imbalance in classification tasks using SMOTE, a widely-used technique in machine learning.
