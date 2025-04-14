# ---------------------------------------------
# ReNewTech Solutions - KNN Classification
# Classification Report Heatmap
# ---------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# SSL Fix
ssl._create_default_https_context = ssl._create_unverified_context

# Load dataset
df = pd.read_csv('data/teleCust1000t.csv')

X = df.drop('custcat', axis=1)
y = df['custcat']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Plot
plt.figure(figsize=(10,8))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5)
plt.title('Classification Report Heatmap - KNN Model', fontsize=18)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.figtext(0.99, 0.01, '© ReNewTech Solutions 2025', ha='right', va='bottom', fontsize=10, color='gray')
plt.tight_layout()
plt.savefig('plots/classification_report_heatmap.png')
plt.show()
