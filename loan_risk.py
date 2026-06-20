import pandas as pd
from sklearn.linear_model import LogisticRegression

# Sample dataset
data = pd.DataFrame({
    'income': [3000, 5000, 2000, 7000],
    'loan_default': [0, 0, 1, 0]
})

X = data[['income']]
y = data['loan_default']

model = LogisticRegression()
model.fit(X, y)

print("Prediction for income=4000:", model.predict([[4000]]))
