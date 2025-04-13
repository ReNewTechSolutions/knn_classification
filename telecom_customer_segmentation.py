# -----------------------------------------------
# ReNewTech Solutions - Telecom Customer Segmentation (KNN Classification)
# -----------------------------------------------

# ✅ 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ✅ 2. Load Dataset
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')

# ✅ 3. Preview Data
print("First 5 Rows of Dataset:")
print(df.head())

# ✅ 4. Basic Info
print("\nDataset Shape:", df.shape)
print("\nDataset Columns:", df.columns)
print("\nClass Distribution (custcat):")
print(df['custcat'].value_counts())
