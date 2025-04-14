# -----------------------------------------------
# ReNewTech Solutions - Find Best K for KNN Model
# -----------------------------------------------

# ✅ Fix SSL verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ✅ Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ✅ Load Dataset
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv'
df = pd.read_csv(url)

# ✅ Feature/Target Split
X = df.drop('custcat', axis=1).values
y = df['custcat'].values

# ✅ Normalize features
X = preprocessing.StandardScaler().fit(X).transform(X)

# ✅ Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# ✅ Find Best K
Ks = 15
mean_acc = np.zeros((Ks-1))

for n in range(1, Ks):
    knn = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = knn.predict(X_test)
    mean_acc[n-1] = accuracy_score(y_test, yhat)

# ✅ Plot accuracy vs K
plt.figure(figsize=(10,6))
plt.plot(range(1, Ks), mean_acc, 'bo-')
plt.title('KNN Accuracy vs K Value')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('plots/accuracy_vs_k.png')
plt.show()

# ✅ Print Best K
best_k = mean_acc.argmax() + 1
print(f"\n✅ Best Accuracy of {mean_acc.max():.2f} achieved at k = {best_k}")
