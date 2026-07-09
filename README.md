# US-Income-1994-
This is a ML Project  that predict whether a person's annual income exceeds $50,000  using U.S. Census data from 1994.


```markdown
# 🎯 Adult Income Classification Project

## Overview
A complete end-to-end machine learning pipeline to predict 
whether a person earns more than **$50,000/year** using 
1994 U.S. Census data. The project covers the full data 
science workflow — from raw data cleaning to model evaluation 
and selection.

---

## 📊 Dataset
- **Source:** UCI Machine Learning Repository — Adult Census Income
- **Size:** 48,842 rows × 15 features
- **Target variable:** `income` (`<=50K` or `>50K`)
- **Type:** Binary Classification

---

## 🗂️ Project Structure

```
Phase 1 — Data Loading, Cleaning & Exploration
├── Loading & combining adult.data + adult.test
├── Handling missing values (MNAR analysis)
├── Descriptive statistics & data inspection
└── Exploratory Data Analysis (EDA)
    ├── Numerical features (age, hours-per-week)
    └── Categorical features (8 variables vs income)

Phase 2 — Encoding, Modeling & Evaluation
├── Feature engineering (dropping redundant columns)
├── Encoding (LabelEncoder + pd.get_dummies)
├── Correlation analysis
├── Train/Test split (80/20, stratified)
├── Feature scaling (StandardScaler)
├── Model training (3 models × 2 versions = 6 total)
├── Model evaluation (Accuracy, F1, Recall, AUC)
├── Feature importance & AUC analysis
└── Final model recommendation
```

---

## 🤖 Models Used
| Model | Version | Best For |
|---|---|---|
| Logistic Regression | Original + Balanced | Baseline + High Recall |
| Decision Tree | Original + Balanced | Interpretability |
| Random Forest | Original + Balanced | Overall Performance |

---

## 📈 Key Results

| Model | Accuracy | Recall `>50K` | F1 `>50K` | AUC |
|---|---|---|---|---|
| LR Balanced | 81% | **0.83** | 0.67 | **0.816** |
| RF Balanced | 83% | 0.74 | **0.68** | 0.801 |

### No Single "Best" Model
- **LR Balanced** → best for financial/business applications
  where missing a high earner is costly (**Recall: 0.83**)
- **RF Balanced** → best for academic/research applications
  where overall balance matters more (**F1: 0.68**)

---

## 🔍 Key Findings
- **Marital status** is the strongest predictor 
  (Married-civ-spouse: 44.6% earn >50K vs 4.5% Never-married)
- **Education** shows a clear upward trend with income 
  (correlation: 0.333)
- **Capital gain**, despite 75% of values being zero, 
  is the strongest single signal in Logistic Regression
- **Class imbalance** (76/24) required `class_weight='balanced'` 
  to improve minority class detection
- **Multicollinearity** detected between `relationship` 
  and `marital-status` — both capture similar information

---

## 🛠️ Tools & Libraries
```python
pandas        # data manipulation & cleaning
numpy         # numerical operations
matplotlib    # data visualization
seaborn       # statistical visualizations
scikit-learn  # machine learning models & evaluation
```

---

## ⚠️ Limitations
- Data is from **1994** — income patterns have changed
- Sex and race disparities reflect historical inequalities
- No hyperparameter tuning applied
- `class_weight='balanced'` is a simple imbalance fix

---

## 🚀 Future Improvements
- Hyperparameter tuning (`max_depth`, `n_estimators`)
- Try **XGBoost** for potentially higher performance
- Apply **log transformation** to `capital-gain`/`capital-loss`
- Formal **fairness audit** across race and sex groups
- **SMOTE** for more sophisticated imbalance handling

---

## 📁 Files
```
├── income analysis.ipynb    # main notebook
├── adult.data               # training data (UCI)
├── adult.test               # test data (UCI)
└── README.md                # this file
```

---

## 👤 Author
- Project completed as part of a data science 
  learning journey
- Dataset: UCI Adult Census Income (1994)
- Source: https://archive.ics.uci.edu/dataset/2/adult
```

This gives your GitHub repository a professional, complete README that covers everything someone needs to know before diving into your notebook. Want me to adjust the author section or add anything specific?
