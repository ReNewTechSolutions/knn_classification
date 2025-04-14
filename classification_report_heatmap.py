p# ---------------------------------------------
# ReNewTech Solutions - KNN Classification Lab
# Classification Report Heatmap Plotter
# ---------------------------------------------

# ✅ Fix SSL issues for loading datasets (Mac-specific)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ✅ Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Scikit-learn tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ✅ Load dataset
df = pd.read_csv('data/teleCust1000t.csv')

# ✅ Prepare features and labels
X = df.drop('custcat', axis=1).values
y = df['custcat'].values

# ✅ Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# ✅ Train KNN model
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# ✅ Generate classification report dictionary
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# ✅ Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5)
plt.title('Classification Report Heatmap (KNN Model)', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# ✅ Save the plot
plt.savefig('plots/classification_report_heatmap.png')
print("✅ Heatmap plot saved to 'plots/classification_report_heatmap.png'.")
plt.show()

# ✅ End of script
