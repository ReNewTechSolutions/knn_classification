![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange.svg)
![Last Updated](https://img.shields.io/badge/Last%20Updated-April%202025-blueviolet)

[![GitHub Project](https://img.shields.io/badge/View%20Project-on%20GitHub-blue?logo=github)](https://github.com/ReNewTechSolutions/knn_classification)

# 📊 KNN Classification - Telecom Customer Segmentation

---

## 📌 Project Overview

This project applies **K-Nearest Neighbors (KNN)** classification to segment telecom customers based on demographic data.  
We predict customer service categories: Basic, E-Service, Plus, or Total Service.

Model evaluation includes:
- Accuracy
- Confusion Matrix
- Visualization of model performance

---

## 📂 Project Structure

knn_classification/ │── README.md │── requirements.txt │── telecom_customer_segmentation.py │── find_best_k.py │── plots/ │── confusion_matrix_knn.png │── accuracy_vs_k.png

---

## 📈 Dataset

- **Source:** IBM Developer Skills Network (Machine Learning with Python specialization)
- **Records:** 1000 customer records with demographics and service usage patterns
- **Target Variable:** `custcat` (customer service category)

---

## 🚀 How to Run This Project

1️⃣ **Clone the repository**
git clone https://github.com/ReNewTechSolutions/knn_classification.git
cd knn_classification
2️⃣ Install dependencies


pip install -r requirements.txt
3️⃣ Train the KNN model


python telecom_customer_segmentation.py
4️⃣ Find the optimal K value

python find_best_k.py
✨ Enhancements Beyond Original Lab
Modular Python scripting for scalability

Automated K optimization (find_best_k.py)

Accuracy vs K plot for K-value tuning

Future upgrade: GridSearchCV hyperparameter tuning

Inspired by IBM coursework and independently extended by ReNewTech Solutions for real-world application readiness.

📌 Author
Felicia Goad | ReNewTech Solutions
Licensed under the MIT License.

🔗 Official Tagline
"Smarter Customer Insights Through Machine Learning — ReNewTech Solutions 2025"