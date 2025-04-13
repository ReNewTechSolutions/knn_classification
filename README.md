# 🔵 K-Nearest Neighbors (KNN) Classification Project

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-brightgreen.svg)
![Last Updated](https://img.shields.io/badge/Last%20Updated-April%202025-blueviolet)

[![GitHub Project](https://img.shields.io/badge/View%20Project-on%20GitHub-blue?logo=github)](https://github.com/ReNewTechSolutions/knn_classification)

---

## 📌 Project Overview

This project applies the **K-Nearest Neighbors (KNN)** algorithm to classify data points from a real-world dataset.  
We explore the effects of different values of `k`, visualize decision boundaries, and evaluate model performance using accuracy and classification metrics.

> 📌 **Inspired by coursework from the IBM Developer Skills Network**, extended independently by ReNewTech Solutions to include enhanced visualizations and fine-tuning experiments.

---

## 📂 Project Structure

knn_classification/ ├── README.md # Project overview and documentation ├── requirements.txt # Required dependencies ├── train_knn.py # Train and evaluate KNN models ├── visualize_results.py # Plot decision boundaries and confusion matrices ├── data/ # Dataset files ├── plots/ # Generated plots and visualizations

yaml
Copy
Edit

---

## 📈 Models Used

- **K-Nearest Neighbors (KNN)** Classifier
- Evaluations across multiple `k` values (e.g., 3, 5, 7, 9)

---

## 🎯 Extra Enhancements (Beyond Course Material)

- Automated plotting of **decision boundaries** across different `k`.
- Confusion Matrix heatmaps for better model evaluation.
- Analysis of how `k` selection impacts model bias and variance.

---

## 🚀 How to Run This Project

1. **Clone the Repository:**

```bash
git clone https://github.com/ReNewTechSolutions/knn_classification.git
cd knn_classification
Install Dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Train Models:

bash
Copy
Edit
python train_knn.py
Visualize Results:

bash
Copy
Edit
python visualize_results.py
🛠 Future Improvements
Explore distance weighting options for KNN

Optimize k selection using cross-validation

Introduce PCA or t-SNE dimensionality reduction for visualization

📌 Author
Felicia Goad
ReNewTech Solutions | 2025

🔗 Official Tagline:
Empowering Smarter Decisions Through Data-Driven Solutions — ReNewTech Solutions 2025.