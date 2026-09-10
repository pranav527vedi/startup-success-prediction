import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"c:\Users\PRANAV\OneDrive\Desktop\project\archive.zip")

df = df.drop(columns=['Unnamed: 0', 'Unnamed: 6', 'id', 'object_id', 'closed_at'])
print(df.shape)
print(df.columns.tolist())


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
df[['age_first_milestone_year', 'age_last_milestone_year']] = imputer.fit_transform(
    df[['age_first_milestone_year', 'age_last_milestone_year']]
)

print(df[['age_first_milestone_year', 'age_last_milestone_year']].isna().sum())

df['status'] = df['status'].map({'acquired': 1, 'closed': 0})
print(df['status'].unique())
print(df['status'].isna().sum())

# ---- Step 4: Remove Redundant Features (labels added here!) ----
df = df.drop(columns=['category_code', 'state_code', 'state_code.1', 'name',
                       'city', 'latitude', 'longitude', 'zip_code',
                       'founded_at', 'first_funding_at', 'last_funding_at',
                       'labels'])

print(df.columns.tolist())

sns.countplot(x='status', data=df)
plt.title('Acquired vs Closed')
##plt.show()

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm')
##plt.show()

X = df.drop(columns=['status'])
y = df['status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled[:2])
print(X_train_scaled.shape, X_test_scaled.shape)


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

from xgboost import XGBClassifier
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate(name, y_true, y_pred):
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print()

evaluate("Logistic Regression", y_test, y_pred_log)
evaluate("Random Forest", y_test, y_pred_rf)
evaluate("XGBoost", y_test, y_pred_xgb)