# 🛡️ AI-Powered Phishing URL Detection and Risk Analysis System

## 📌 Project Overview

The AI-Powered Phishing URL Detection and Risk Analysis System is a machine learning project designed to classify website records as phishing or legitimate using website-related features.

The project compares multiple machine learning algorithms and provides an interactive dashboard built using Streamlit.

## 🎯 Objectives

- Detect phishing-related website patterns using machine learning.
- Compare Logistic Regression, Decision Tree, and Random Forest.
- Evaluate model performance using standard classification metrics.
- Present predictions and model comparisons through an interactive dashboard.

## 🧰 Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Streamlit
- **Model Storage:** Joblib
- **Dataset:** UCI Phishing Websites dataset

## 🤖 Machine Learning Models

Three classification algorithms are evaluated:

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 92.85% |
| Decision Tree | 97.11% |
| Random Forest | 97.42% |

Random Forest achieved the highest accuracy on the current 20% held-out test split.

## 📊 Dataset

The dataset contains 11,055 website records and 30 input features. The target labels are:

- `-1`: Phishing website
- `1`: Legitimate website

The model uses the dataset's precomputed features. Direct classification of arbitrary raw URLs is not yet implemented.

## ✨ Features

- Automated model training and comparison
- Saved trained Random Forest model
- Interactive Streamlit dashboard
- Dataset-based prediction demonstration
- Model comparison chart
- Classification report with precision, recall, and F1-score

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd phishing-url-detection
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install