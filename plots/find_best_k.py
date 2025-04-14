# ---------------------------------------------
# find_best_k.py - ReNewTech Solutions
# ---------------------------------------------

# ✅ TEMP SSL Fix for MacOS / Python 3.13
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ✅ Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ✅ Load dataset
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')

# ✅ Features and Labels
X = df.drop('custcat', axis=1).values
y = df['custcat'].values

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ✅ Find best K
k_range = range(1, 30)
accuracies = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    accuracies.append(acc)

# ✅ Plot Accuracy vs K
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracies, marker='o')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs K Value - KNN Model')
plt.grid()
plt.savefig('plots/accuracy_vs_k.png')
plt.show()
