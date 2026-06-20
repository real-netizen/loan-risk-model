import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -----------------------------
# 1. Create a sample dataset
# -----------------------------
data = pd.DataFrame({
    'income': [2500, 4000, 1500, 7000, 3000, 8000, 2000, 5000],
    'credit_score': [580, 720, 500, 800, 650, 750, 550, 700],
    'loan_amount': [1000, 5000, 2000, 10000, 3000, 12000, 1500, 6000],
    'employment_years': [1, 5, 0, 10, 3, 12, 2, 7],
    'default': [1, 0, 1, 0, 0, 0, 1, 0]  # 1 = default, 0 = repaid
})

X = data[['income','credit_score','loan_amount','employment_years']]
y = data['default']

# -----------------------------
# 2. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# 3. Train model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# 4. Evaluate
# -----------------------------
y_pred = model.predict(X_test)
print("Loan Risk Assessment Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 5. Example prediction
# -----------------------------
sample = pd.DataFrame({
    'income':[4500],
    'credit_score':[680],
    'loan_amount':[4000],
    'employment_years':[4]
})

prediction = model.predict(sample)
print("Prediction for sample applicant:", "DEFAULT RISK" if prediction[0]==1 else "LOW RISK")
