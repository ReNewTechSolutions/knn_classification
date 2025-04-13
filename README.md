# K-Nearest Neighbors (KNN) Classification Project

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-brightgreen.svg)
![Last Updated](https://img.shields.io/badge/Last%20Updated-April%202025-blueviolet)
[![GitHub Project](https://img.shields.io/badge/View%20Project-on%20GitHub-blue?logo=github)](https://github.com/ReNewTechSolutions/knn_classification)

---

## 📌 Project Overview

This project applies the **K-Nearest Neighbors (KNN)** algorithm to real-world datasets for **classification** tasks.

We explore:
- The impact of different values of `k`
- Model evaluation through ROC-AUC, Accuracy, and F1-Score
- Visualization of decision boundaries
- Practical applications in customer segmentation

Developed by **ReNewTech Solutions** to expand applied machine learning expertise.

---

## 📂 Project Structure

knn_classification/ │ ├── README.md — Project overview and documentation ├── requirements.txt — Required Python dependencies │ ├── train_knn.py — Train KNN model on synthetic data ├── visualize_results.py — Generate decision boundary plots │ ├── telecom_customer_segmentation.py — KNN model on telecom real-world dataset │ ├── data/ — Data files (if needed) └── plots/ — Saved visualizations and ROC curves


---

## 📊 Datasets Used

- **Synthetic classification datasets** for boundary visualization
- **Telecom Customer Dataset** (IBM Developer Skills Network)  
  Focus: Segment telecom customers based on demographic and service usage patterns

---

## ✨ Key Enhancements

- Expanded original IBM lab to include:
  - **ROC Curve visualization**
  - **Feature scaling**
  - **Model evaluation comparison (Accuracy, F1, ROC-AUC)**
- Built modular code for clean training and evaluation scripts
- Professional repository and project organization under **ReNewTech Solutions**

---

## 🚀 How to Run This Project

1️⃣ Clone the repository

```bash
git clone https://github.com/ReNewTechSolutions/knn_classification.git
cd knn_classification
2️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run training scripts

bash
Copy
Edit
python train_knn.py
python telecom_customer_segmentation.py
4️⃣ Visualize results

bash
Copy
Edit
python visualize_results.py
🛠 Future Improvements
Hyperparameter tuning (automated k optimization)

Expand to multi-class datasets with imbalance

KNN for regression tasks

📌 Author
Felicia Goad, ReNewTech Solutions

Licensed under MIT License

🔗 Official Tagline:
Expanding Practical AI Solutions — One Classification at a Time.