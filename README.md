<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue.svg" alt="Python Badge">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License Badge">
  <img src="https://img.shields.io/badge/Status-In%20Progress-orange.svg" alt="Status Badge">
  <img src="https://img.shields.io/badge/Last%20Updated-April%202025-blueviolet" alt="Last Updated Badge">
  <br/>
  <a href="https://github.com/ReNewTechSolutions/knn_classification">
    <img src="https://img.shields.io/badge/View%20Project-on%20GitHub-blue?logo=github" alt="GitHub Project Badge">
  </a>
</p>

# 📊 KNN Classification - Telecom Customer Segmentation

---

## 📌 Project Overview

This project applies **K-Nearest Neighbors (KNN)** classification to segment telecom customers based on demographic data.  
We predict customer service categories: **Basic**, **E-Service**, **Plus**, or **Total Service**.

Model evaluation includes:
- Accuracy
- Confusion Matrix
- Visualizations

---

## 📂 Project Structure

knn_classification/ │── README.md │── requirements.txt │── telecom_customer_segmentation.py │── find_best_k.py │── visualize_results.py │── classification_report_heatmap.py │── plots/ │ ├── confusion_matrix_knn.png │ ├── accuracy_vs_k.png │ ├── correlation_matrix.png │ └── classification_report_heatmap.png

---

## 📈 Dataset

- **Source:** IBM Developer Skills Network (Machine Learning with Python specialization)
- **Records:** 1000 customer records
- **Features:** Demographics and service usage
- **Target:** `custcat` (customer service category)

---

## 🚀 How to Run This Project

1️⃣ **Clone the repository**

```bash
git clone https://github.com/ReNewTechSolutions/knn_classification.git
cd knn_classification
2️⃣ Install dependencies


pip install -r requirements.txt
3️⃣ Train the KNN model


python telecom_customer_segmentation.py
4️⃣ Find the optimal K value


python find_best_k.py
5️⃣ Generate Classification Report Heatmap


python classification_report_heatmap.py
6️⃣ Visualize Results


python visualize_results.py
📸 Generated Visuals
Confusion Matrix for KNN Predictions

Accuracy vs K Value Plot

Correlation Heatmap of Features

Classification Report Heatmap

(Visualizations are saved automatically in the /plots/ directory.)

✨ Enhancements Beyond Original Lab
Modular Python scripting for scalability

Automated K optimization (find_best_k.py)

Correlation Matrix Heatmap added

Professional GitHub-ready structure

SSL issues fixed for Mac OS / Python 3.13+ environments

Future upgrade planned: GridSearchCV hyperparameter tuning

Inspired by IBM coursework, independently extended by ReNewTech Solutions for real-world application readiness.

📌 Author
Felicia Goad | ReNewTech Solutions
Licensed under the MIT License.

🔗 Official Tagline
"Smarter Customer Insights Through Machine Learning — ReNewTech Solutions 2025"