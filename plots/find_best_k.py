# ---------------------------------------------
# ReNewTech Solutions - KNN Classification
# Finding Best K
# ---------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ssl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# SSL Fix
ssl._create_default_https_context = ssl._create_unverified_context

# Load dataset
df = pd.read_csv('data/teleCust1000t.csv')

X = df.drop('custcat', axis=1)
y = df['custcat']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

k_range = range(1, 20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

# Plot
plt.figure(figsize=(10,6))
plt.plot(k_range, scores, marker='o')
plt.title('Accuracy vs. K Value', fontsize=18)
plt.xlabel('K', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(k_range)
plt.grid(True)
plt.figtext(0.99, 0.01, '© ReNewTech Solutions 2025', ha='right', va='bottom', fontsize=10, color='gray')
plt.tight_layout()
plt.savefig('plots/accuracy_vs_k.png')
plt.show()
