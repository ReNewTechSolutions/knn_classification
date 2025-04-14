# ---------------------------------------------
# ReNewTech Solutions - KNN Classification
# Enhanced Visualization: Confusion Matrix + Correlation Matrix
# ---------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# SSL Fix for Mac
ssl._create_default_https_context = ssl._create_unverified_context

# Load dataset
df = pd.read_csv('data/teleCust1000t.csv')

# Features and labels
X = df.drop('custcat', axis=1)
y = df['custcat']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5)
plt.title('Confusion Matrix - KNN Model', fontsize=18)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.figtext(0.99, 0.01, '© ReNewTech Solutions 2025', ha='right', va='bottom', fontsize=10, color='gray')
plt.tight_layout()
plt.savefig('plots/confusion_matrix_knn.png')
plt.show()

# Correlation Matrix
correlation_matrix = df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=18)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.figtext(0.99, 0.01, '© ReNewTech Solutions 2025', ha='right', va='bottom', fontsize=10, color='gray')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png')
plt.show()
