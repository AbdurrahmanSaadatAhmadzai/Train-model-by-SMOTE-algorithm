import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

# Load the imbalanced dataset
imbalanced_data = pd.read_csv('/Users/student/Desktop/ML/imbalanced_dataset.csv')


print("Imbalanced Dataset:")
print(imbalanced_data.head())

# Plot class distribution of imbalanced dataset
plt.figure(figsize=(8, 6))
plt.bar(imbalanced_data['target'].unique(), imbalanced_data['target'].value_counts(), color=['blue', 'red'])
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Class Distribution of Imbalanced Dataset')
plt.xticks([0, 1], ['Non-Seizure', 'Seizure'])
plt.show()

# Split the imbalanced data into features (X) and target variable (y)
X_imbalanced = imbalanced_data.drop('target', axis=1)
y_imbalanced = imbalanced_data['target']

# Split the imbalanced data into training and testing sets
X_train_imbalanced, X_test_imbalanced, y_train_imbalanced, y_test_imbalanced = train_test_split(X_imbalanced, y_imbalanced, test_size=0.2, random_state=42)

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to training data only
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imbalanced, y_train_imbalanced)

# Convert resampled data to DataFrame
df_resampled = pd.DataFrame(X_train_resampled, columns=[f'feature_{i}' for i in range(X_train_imbalanced.shape[1])])
df_resampled['target'] = y_train_resampled

# Display the balanced dataset after SMOTE
print("\nBalanced Dataset after SMOTE:")
print(df_resampled.head())

# Plot class distribution of balanced dataset
plt.figure(figsize=(8, 6))
plt.bar(df_resampled['target'].unique(), df_resampled['target'].value_counts(), color=['blue', 'red'])
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Class Distribution of Balanced Dataset after SMOTE')
plt.xticks([0, 1], ['Non-Seizure', 'Seizure'])
plt.show()

# Train a Random Forest classifier using the balanced data after SMOTE
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test_imbalanced)

# Evaluate the model
accuracy = accuracy_score(y_test_imbalanced, y_pred)
print('\nModel Evaluation on Imbalanced Test Data:')
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test_imbalanced, y_pred))
