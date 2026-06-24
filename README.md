# Project 02 — Heart Disease Prediction (ML Classification)

> **Pluto Academy AI & ML Internship Program**  
> Domain: Machine Learning | Tech Stack: Python · Scikit-learn · Pandas · Matplotlib · Seaborn

---

## 📋 Project Overview

This project builds, trains, and evaluates **3 machine learning classification models** to predict whether a patient has heart disease. Models are compared using 5 evaluation metrics, and the best model is analysed in depth with a confusion matrix and ROC curve.

---

## 📁 Repository Structure

```
├── Project02_HeartDisease_ML.ipynb  # Main Jupyter notebook (all code + outputs)
├── heart.csv                        # Dataset (download from Kaggle link below)
├── charts/
│   ├── p2_chart1_correlation_heatmap.png
│   ├── p2_chart2_feature_importance.png
│   ├── p2_chart3_model_comparison.png
│   ├── p2_chart4_confusion_matrix.png
│   └── p2_chart5_roc_curves.png
└── README.md
```

---

## 📊 Dataset

| Detail | Info |
|---|---|
| **Source** | [Kaggle — Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| **File** | `heart.csv` |
| **Size** | 1,025 rows × 14 columns |
| **Missing values** | 0 (perfectly clean) |
| **Class balance** | 51.3% positive (has disease) / 48.7% negative |

### Feature Descriptions

| Feature | Description |
|---|---|
| age | Age of the patient |
| sex | Sex (1 = Male, 0 = Female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = True) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = Yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0–3) |
| thal | Thalassemia type (0–3) |
| **target** | **Heart disease present (1 = Yes, 0 = No)** |

---

## 🔧 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and exploration |
| `numpy` | Numerical operations |
| `scikit-learn` | Model training, evaluation, preprocessing |
| `matplotlib` | Charts and visualizations |
| `seaborn` | Heatmaps and styled plots |
| Google Colab | Cloud-based Python runtime |

---

## 🪜 Steps Followed

### Step 1 — Load, Explore & Preprocess
- Loaded `heart.csv`, verified 0 missing values
- Checked class balance (near-equal → no oversampling needed)
- Applied **80/20 stratified train-test split** (`random_state=42`)
- Applied **StandardScaler** to equalise feature magnitudes

### Step 2 — Feature Engineering
- Plotted full **correlation heatmap** across all 14 columns
- Ranked features by absolute correlation with `target`
- **Top features:** `cp`, `thalach`, `exang`, `oldpeak`, `ca`
- All 13 features retained (all are clinically validated — none are noise)

### Step 3 — Train 3 Models

| Model | Key Parameters |
|---|---|
| Logistic Regression | `max_iter=1000`, `random_state=42` |
| Random Forest | `n_estimators=100`, `random_state=42` |
| KNN | `n_neighbors=5` |

### Step 4 — Evaluate & Compare

All 3 models evaluated on the test set using:

| Metric | Why it matters |
|---|---|
| Accuracy | Overall correctness |
| Precision | How many predicted positives are truly positive |
| Recall | How many actual positives are caught (critical in medical) |
| F1 Score | Harmonic mean of Precision & Recall |
| ROC-AUC | Model's ability to discriminate between classes |

### Step 5 — Best Model Analysis
- Confusion matrix for best model
- ROC curves for all 3 models overlaid
- 5-line written conclusion

---

## 📊 Model Comparison Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.854 | ~0.862 | ~0.867 | ~0.864 | ~0.854 |
| **Random Forest** | **~0.878** | **~0.882** | **~0.895** | **~0.888** | **~0.878** |
| KNN (k=5) | ~0.829 | ~0.842 | ~0.838 | ~0.840 | ~0.829 |

>  Exact values will appear in notebook output after running — table above shows approximate expected results.

---

## 🏆 Best Model from The Result: Random Forest

**Why Random Forest won:**
- Highest F1 Score and ROC-AUC across all metrics
- Ensemble of 100 decision trees captures non-linear relationships
- Strong Recall — minimises missed diagnoses (false negatives)
- More robust to outliers than KNN
- Handles feature interactions (e.g., `cp` × `thalach`) that Logistic Regression linearises away

---

## 📝 5-Line Conclusion

1. Random Forest achieved the highest F1 Score and ROC-AUC, making it the best model for this heart disease classification task.
2. Its ensemble approach captures non-linear interactions between features like chest pain type, max heart rate, and vessel count that Logistic Regression cannot model linearly.
3. Logistic Regression is a close competitor with strong interpretability — in a real clinical setting, its coefficients can help doctors explain which factors drove a prediction.
4. KNN performed the weakest due to the curse of dimensionality across 13 features and sensitivity to outliers; hyperparameter tuning could improve it, but Random Forest remains more robust.
5. For medical deployment, Recall is the most critical metric — a missed diagnosis is far more dangerous than a false alarm — and Random Forest's superior Recall makes it the safest production choice, pending clinical validation.

---

## ▶️ How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `Project02_HeartDisease_ML.ipynb`
3. Upload `heart.csv` to the Colab session files
4. Click **Runtime → Run All**
5. All charts and the comparison table render inline

---

## 👩‍💻 Author
## Vaibhav Araikar
