# Student Performance Prediction System Using Machine Learning

An end-to-end, production-ready **Supervised Machine Learning Web Application** designed for predicting student academic performance (`Good`, `Average`, `Poor`) based on academic indicators, attendance trends, continuous assessments, and lifestyle habits.

Built for college **Machine Learning Mini-Project** demonstrations and viva evaluations, featuring full preprocessing pipelines, 4 classification algorithms, interactive Chart.js visualizations, SQLite history tracking, custom CSV uploads, and explainable data-driven insights.

---

## 📌 Project Overview

- **Project Title**: Student Performance Prediction System Using Machine Learning
- **Target Variable**: `Performance_Category` (`Poor` = 0, `Average` = 1, `Good` = 2)
- **Primary Tech Stack**: Python, Flask, Scikit-Learn, Pandas, NumPy, HTML5/CSS3 (Tailwind CSS), Chart.js, SQLite
- **Location**: `C:\Users\Asus\.gemini\antigravity\scratch\student-performance-ml`

---

## 🚀 Key Features

1. **Synthetic & Custom Dataset Support**:
   - Includes realistic 500-record sample dataset (`student_performance.csv`).
   - Allows users to upload their own `.csv` dataset dynamically.
2. **Robust Preprocessing Pipeline**:
   - Automated missing value imputation (median for numeric, mode for categorical).
   - Duplicate record purging and range outlier validation.
   - Categorical encoding (`Extracurricular_Activities`: Yes=1, No=0).
   - Standardization via `StandardScaler` fitted on training set only (preventing data leakage).
   - Stratified 80/20 train-test split (`random_state=42`).
3. **Exploratory Data Analysis (EDA)**:
   - Interactive Chart.js visualizations for class distributions, attendance vs marks, study hours trends, feature histograms, and Pearson correlation heatmaps.
4. **Multi-Model Machine Learning Training**:
   - Trains and compares **4 Classification Algorithms**:
     1. Logistic Regression
     2. Decision Tree Classifier
     3. Random Forest Classifier
     4. K-Nearest Neighbors (KNN)
   - Model binaries saved to disk (`models/trained_models/*.joblib`).
5. **Comprehensive Evaluation**:
   - Calculates Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1-Score, and 3x3 Confusion Matrices on held-out testing data.
   - Horizontal bar charts for Tree feature importances.
6. **Live Prediction & Data-Driven Insights**:
   - Interactive prediction form with preset profiles (High Achiever, Average, At-Risk).
   - Real-time confidence meter percentage and probability breakdown.
   - Rule-based and model-based feature influence explanations highlighting attendance, backlog, and study habit impacts.
7. **SQLite History Logging**:
   - Automatically logs all predictions to `database/predictions.db`.
   - Includes option to view, search, and clear prediction logs.

---

## 🛠️ Technology Stack

| Layer | Technologies Used |
| :--- | :--- |
| **Frontend** | HTML5, CSS3, JavaScript, Tailwind CSS (via CDN), Font Awesome 6, Chart.js |
| **Backend API** | Python 3.14 / 3.10+, Flask, Jinja2 |
| **Machine Learning** | Scikit-Learn, Pandas, NumPy, Joblib, Matplotlib, Seaborn |
| **Database** | SQLite3 (`predictions.db`) |

---

## 📁 Project Directory Structure

```
student-performance-ml/
│
├── app.py                      # Flask Application Controller & REST Endpoints
├── requirements.txt            # Python Package Dependencies
├── README.md                   # Complete Documentation & Setup Guide
├── test_ml_pipeline.py         # Standalone Automated Test Suite for ML Core
├── test_app_routes.py          # Automated Flask Route & API Test Suite
│
├── dataset/
│   ├── generate_dataset.py     # Synthetic Dataset Generator Script
│   └── student_performance.csv # 500-Row Sample Dataset
│
├── models/
│   └── trained_models/         # Joblib Binary Files (LR, DT, RF, KNN, Scaler)
│
├── preprocessing/
│   └── preprocess.py           # Preprocessing & StandardScaler Pipeline
│
├── ml/
│   ├── train.py                # Model Training Core (LR, DT, RF, KNN)
│   ├── evaluate.py             # Accuracy, Precision, Recall, F1 & Confusion Matrix
│   └── predict.py              # Inference Engine & Feature Impact Insights
│
├── database/
│   ├── db.py                   # SQLite Database Handler
│   └── predictions.db          # History Database File
│
├── templates/
│   ├── layout.html             # Base Navigation Layout Template
│   ├── index.html              # Home Page & Workflow Visualizer
│   ├── dataset.html            # Dataset Upload, Preview & Statistics
│   ├── preprocessing.html      # Data Preprocessing Summary & Split Info
│   ├── eda.html                # Exploratory Data Analysis & Charts
│   ├── training.html           # Model Training Control & Comparison
│   ├── evaluation.html         # Evaluation Metrics & Confusion Matrices
│   ├── prediction.html         # Interactive Prediction Form & Insights
│   ├── history.html            # SQLite Log Viewer
│   └── about.html              # Algorithm Mechanics & Viva Preparation
│
└── static/
    ├── css/
    │   └── style.css           # Custom Glassmorphism Styling
    └── js/
        ├── main.js             # Form & AJAX Event Logic
        └── charts.js           # Chart.js Initialization Script
```

---

## ⚙️ Installation & Running Locally

### Step 1: Clone / Navigate to Directory
```bash
cd C:\Users\Asus\.gemini\antigravity\scratch\student-performance-ml
```

### Step 2: Set Up Virtual Environment & Dependencies
```bash
# Using standard Python venv
python -m venv .venv

# Activate Virtual Environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### Step 3: Launch Flask Application
```bash
python app.py
```

Open your browser and navigate to:
`http://127.0.0.1:5000`

---

## 🔗 REST API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /` | GET | Renders Home Dashboard |
| `POST /upload-dataset` | POST | Uploads custom CSV dataset |
| `GET /dataset-info` | GET | Returns dataset shape, column types, preview, missing values |
| `POST /preprocess` | POST | Runs preprocessing pipeline & returns clean summary |
| `GET /eda-data` | GET | Returns EDA chart data (distributions, correlations, scatter) |
| `POST /train` | POST | Trains all 4 ML models, saves `.joblib` files, returns metrics |
| `GET /evaluation` | GET | Returns accuracy, precision, recall, F1, and confusion matrices |
| `POST /predict` | POST | Accepts student feature JSON, returns prediction & insights |
| `GET /prediction-history` | GET | Returns logged prediction records from SQLite |
| `DELETE /prediction-history`| DELETE | Clears prediction table in SQLite |

---

## 📊 Model Performance Comparison

Evaluated on held-out 20% test dataset ($N=100$):

| Algorithm | Model Type | Test Accuracy | F1-Score (Weighted) | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression** | Linear Softmax | **90.0%** | **90.0%** | Trained & Saved |
| **Random Forest** | Ensemble Bagging | **84.0%** | **84.0%** | Trained & Saved |
| **Decision Tree** | Binary Splits | **80.0%** | **80.1%** | Trained & Saved |
| **K-Nearest Neighbors** | Instance Distance ($k=5$) | **76.0%** | **76.0%** | Trained & Saved |

---

## 🎓 College Viva Evaluation Questions & Answers

### Q1: What is the main objective of this Machine Learning project?
**Answer:** To build a supervised classification application that early-identifies student academic standing (`Good`, `Average`, `Poor`) based on attendance, study hours, past marks, and lifestyle factors, allowing timely academic interventions.

### Q2: Why did you use `StandardScaler` on numerical features?
**Answer:** Distance-based algorithms like KNN and gradient-based models like Logistic Regression are sensitive to feature scales. `StandardScaler` standardizes features to zero mean ($\mu=0$) and unit variance ($\sigma=2=1$), ensuring no single large-scale feature (e.g. Attendance 0-100%) dominates smaller features (e.g. Study Hours 1-10).

### Q3: How did you prevent Data Leakage during preprocessing?
**Answer:** We performed the train-test split *before* fitting the scaler. The `StandardScaler` was fitted strictly on `X_train`, and the resulting fitted object was used to transform `X_test` and incoming single prediction requests.

### Q4: Explain the difference between Precision and Recall in this project context.
**Answer:** 
- **Precision**: Of all students predicted as `Poor`, what percentage actually had `Poor` performance? (High precision avoids false alarms).
- **Recall**: Of all students who were *actually* `Poor`, what percentage did the model correctly catch? (High recall ensures no at-risk student is missed).

### Q5: Why is Random Forest generally superior to a single Decision Tree?
**Answer:** Single decision trees easily overfit noisy training data (high variance). Random Forest constructs 100 decorrelated decision trees using bootstrap aggregation (bagging) and random feature selection, averaging predictions to reduce variance while maintaining low bias.

---

## 📷 Screenshots Section Placeholders

- `[Screenshot Placeholder 1: Home Dashboard & Interactive Workflow]`
- `[Screenshot Placeholder 2: Dataset Upload & Summary Statistics]`
- `[Screenshot Placeholder 3: Exploratory Data Analysis Charts]`
- `[Screenshot Placeholder 4: Model Evaluation Matrix & Confusion Grid]`
- `[Screenshot Placeholder 5: Live Prediction Form & Insights Card]`

---

## 🔮 Future Scope

1. **Deep Learning Integration**: Incorporating Multi-Layer Perceptron (MLP) neural networks for non-linear feature interaction modeling.
2. **LMS & ERP Integration**: Connecting directly via REST APIs to University Learning Management Systems (Canvas / Moodle) to fetch real-time attendance and assignment submissions.
3. **Automated Remedial Recommendations**: Generating tailored study timetables and learning resource recommendations based on identified weakness areas.
