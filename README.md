# startup-success-prediction
# 🚀 Startup Success Prediction

Predicting whether a startup will be **acquired** or will **shut down**, using funding history, investor activity, location, and milestone data — a binary classification project built end-to-end with pandas, scikit-learn, and XGBoost.

---

## 📌 Problem Statement

Startups either succeed (get acquired) or fail (shut down). Using structured historical data — funding rounds, total capital raised, investor participation, location, category, and milestones — this project builds ML models to predict that outcome.

**Target variable:** `status`
- `1` → Acquired
- `0` → Closed

---

## 📊 Dataset

- **923 startups**, 49 raw columns → cleaned down to 32 usable features
- **Source:** [Startup Success Prediction — Kaggle](https://www.kaggle.com/datasets/manishkc06/startup-success-prediction)
- **Class balance:** 597 acquired vs 326 closed (moderately imbalanced)

---

## 🛠 Pipeline

| Step | What was done |
|---|---|
| **1. Data Cleaning** | Dropped identifier columns (`id`, `object_id`, `Unnamed: 0`) and `closed_at` (leaked the target directly) |
| **2. Missing Values** | Median imputation on `age_first_milestone_year`, `age_last_milestone_year` — robust to outliers |
| **3. Target Encoding** | Mapped `status` → binary (`acquired`=1, `closed`=0) via `.map()` |
| **4. Redundant Features** | Removed `category_code`, `state_code` (already one-hot encoded elsewhere), raw location/date columns, and `labels` (found to be an exact duplicate of the target) |
| **5. EDA** | Class balance countplot + correlation heatmap |
| **6. Train-Test Split** | 80/20 stratified split to preserve class ratio |
| **7. Feature Scaling** | `StandardScaler`, fit on train only (avoids leakage into test set) |
| **8. Model Training** | Logistic Regression, Random Forest, XGBoost |
| **9. Evaluation** | Accuracy, Precision, Recall, F1, Confusion Matrix |

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.730 | 0.792 | 0.792 | 0.792 |
| Random Forest | **0.784** | 0.817 | **0.858** | **0.837** |
| XGBoost | 0.757 | 0.800 | 0.833 | 0.816 |

**Best model: Random Forest** — highest accuracy and F1-score among the three.

### Confusion Matrices

**Logistic Regression**
```
[[40 25]
 [25 95]]
```

**Random Forest**
```
[[42 23]
 [17 103]]
```

**XGBoost**
```
[[40 25]
 [20 100]]
```

---

## 🐛 Key Debugging Story: Catching Data Leakage

While debugging, all three models initially returned a suspicious **100% accuracy** — a strong red flag rather than good news. Investigation revealed a `labels` column that was an **exact duplicate of the target `status` column**, silently leaking the answer into the feature set. Dropping it and re-training produced realistic, trustworthy metrics.

> This is a good reminder: perfect scores on real-world data usually mean a bug, not a breakthrough.

---

## 🧰 Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost

---

## ▶️ How to Run

```bash
git clone <your-repo-url>
cd startup-success-prediction
pip install -r requirements.txt
python "startup success prediction.py"
```

---

## 📁 requirements.txt

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

---

## 🔮 Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Feature importance analysis to identify top predictors
- Handle class imbalance with SMOTE or class weighting
- Try additional models (LightGBM, CatBoost)
