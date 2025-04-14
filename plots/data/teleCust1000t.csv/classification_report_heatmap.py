import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load your dataset (local)
df = pd.read_csv('data/teleCust1000t.csv')

# 2. Preprocessing
X = df.drop('custcat', axis=1)
y = df['custcat']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 4. Generate classification report
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# 5. Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
plt.title('Classification Report Heatmap - KNN Model')
plt.tight_layout()
plt.savefig('plots/classification_report_heatmap.png')
plt.show()
