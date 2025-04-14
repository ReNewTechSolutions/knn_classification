# -----------------------------------------------
# ReNewTech Solutions - Telecom Customer Segmentation (KNN Classification)
# -----------------------------------------------

# ✅ Fix SSL verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ✅ Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ✅ Load Dataset
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv'
df = pd.read_csv(url)

print("🚀 Dataset loaded successfully!\n")
print(df.head())

# ✅ Quick Explore
print("\n🔍 Customer Category Value Counts:\n")
print(df['custcat'].value_counts())

# ✅ Define features (all except 'custcat')
X = df.drop('custcat', axis=1).values
y = df['custcat'].values

# ✅ Normalize features
X = preprocessing.StandardScaler().fit(X).transform(X)

# ✅ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# ✅ Model Training
k = 4
knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

# ✅ Predictions
y_pred = knn.predict(X_test)

# ✅ Evaluation
print("\n🏁 Model Evaluation (k=4):\n")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ✅ Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix - KNN (k=4)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('plots/confusion_matrix_knn.png')
plt.show()
