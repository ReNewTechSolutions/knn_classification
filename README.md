# K-Nearest Neighbors (KNN) Classification Project

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-brightgreen.svg)
![Last Updated](https://img.shields.io/badge/Last%20Updated-April%202025-blueviolet)
[![GitHub Project](https://img.shields.io/badge/View%20Project-on%20GitHub-blue?logo=github)](https://github.com/ReNewTechSolutions/knn_classification)

---

## 📌 Project Overview

This project applies the **K-Nearest Neighbors (KNN)** algorithm to classify data points from a real-world dataset.

We explore:
- How different values of `k` affect performance.
- Visualizing decision boundaries.
- Evaluating model accuracy, confusion matrix, and classification metrics.

---

## 🛠 Project Structure

knn_classification/ │ ├── README.md # Project overview and documentation ├── requirements.txt # Required dependencies │ ├── train_knn.py # Train KNN models with different k values ├── visualize_results.py # Visualize decision boundaries and ROC curves │ ├── data/ # (Optional) Folder for datasets if needed │ └── plots/ # Generated plots and decision boundary visualizations

---

## 📊 Dataset

- Real-world dataset for binary classification tasks.
- Data is scaled for better KNN performance.

---

## ✨ Custom Enhancements Beyond Original Lab

This project is inspired by coursework from the **IBM Developer Skills Network** (Machine Learning with Python specialization).

🔵 **Enhancements made independently at ReNewTech Solutions:**
- Added visualizations of decision boundaries across different `k` values.
- Compared classification results across multiple metrics (accuracy, recall, precision, F1-score).
- Professional project structuring for scalability and reuse.

---

## 🚀 How to Run This Project

1️⃣ Clone the repository:

```bash
git clone https://github.com/ReNewTechSolutions/knn_classification.git
cd knn_classification
2️⃣ Install the dependencies:

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Train KNN models:

bash
Copy
Edit
python train_knn.py
4️⃣ Visualize results:

bash
Copy
Edit
python visualize_results.py
📈 Metrics Evaluated
Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Decision Boundaries

🛠 Future Improvements
Automate hyperparameter tuning with GridSearchCV.

Extend to multiclass classification datasets.

Benchmark KNN against other classifiers (SVM, Decision Trees).

👨‍💻 Author
Felicia Goad — ReNewTech Solutions

License: MIT

🔗 Official Tagline
Smarter Classification with Machine Learning — Powered by ReNewTech Solutions, 2025.