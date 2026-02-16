# -*- coding: utf-8 -*-
"""
Created on Sat May  3 10:29:56 2025

@author: lenovo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read and process QCM files
def process_qcm_file(filepath):
    df = pd.read_csv(filepath, header=None)
    columns = df.iloc[0, 0].split(';')
    data_rows = df.iloc[1:, 0].str.split(';', expand=True)
    data_rows.columns = columns
    data_rows = data_rows.apply(pd.to_numeric)
    return data_rows

# List of input files
file_paths = ["QCM3.csv", "QCM6.csv", "QCM7.csv", "QCM10.csv", "QCM12.csv"]

# Combine data from all files
all_data = pd.DataFrame()
for path in file_paths:
    processed_df = process_qcm_file(path)
    all_data = pd.concat([all_data, processed_df], ignore_index=True)

# Separate features (X) and labels (y)
X = all_data.iloc[:, :-5]
y = all_data.iloc[:, -5:]
y_single_label = y.idxmax(axis=1).str.replace('"', '')  # Convert to single label per row

# Normalize the features to aid training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define hyperparameter grid with best alpha only
param_grid = {
    "hidden_layer_sizes": [(100,), (50,), (100, 100), (50, 50), (100, 50)],
    "learning_rate_init": [0.1, 0.01],
    "alpha": [0.0001]  # best value from previous search
}

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform grid search with cross-validation
best_accuracy = 0
best_model = None
best_params = None
results = []

print("Grid Search with Cross-Validation - Results:")
for params in ParameterGrid(param_grid):
    fold_accuracies = []
    for train_index, test_index in skf.split(X_scaled, y_single_label):
        X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
        y_train_fold, y_test_fold = y_single_label.iloc[train_index], y_single_label.iloc[test_index]

        model = MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            learning_rate_init=params["learning_rate_init"],
            alpha=params["alpha"],
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)
        acc = accuracy_score(y_test_fold, y_pred_fold)
        fold_accuracies.append(acc)

    mean_acc = np.mean(fold_accuracies)
    print(f"Layers: {params['hidden_layer_sizes']}, LR: {params['learning_rate_init']}, Alpha: {params['alpha']}, Mean Accuracy: {mean_acc:.4f}")
    results.append((f"{params['hidden_layer_sizes']}, lr={params['learning_rate_init']}", mean_acc))

    if mean_acc > best_accuracy:
        best_accuracy = mean_acc
        best_model = model
        best_params = params

labels, accuracies = zip(*results)
plt.figure(figsize=(10, 6))
plt.bar(labels, accuracies, color='lightblue')
plt.xlabel("Mean Accuracy")
plt.title("Accuracy per Hyperparameter Configuration")
plt.tight_layout()
plt.show()

# Final evaluation on separate test set
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_scaled, y_single_label, test_size=0.25, random_state=42)
best_model.fit(X_train_final, y_train_final)
y_pred_final = best_model.predict(X_test_final)

print("\nBest Configuration:")
print(f"Hidden Layers: {best_params['hidden_layer_sizes']}")
print(f"Learning Rate: {best_params['learning_rate_init']}")
print(f"Alpha: {best_params['alpha']}")
print(f"Final Accuracy: {accuracy_score(y_test_final, y_pred_final):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_final, y_pred_final))