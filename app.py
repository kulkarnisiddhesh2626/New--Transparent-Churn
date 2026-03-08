import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

# Load data
df = pd.read_csv('/Telco-Customer-Churn.csv.csv')

# Clean up
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df.drop('customerID', axis=1, inplace=True)

# Feature Engineering
df['Tenure_to_Monthly_Ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1)
df['New_High_Charge'] = ((df['tenure'] < 6) & (df['MonthlyCharges'] > 70)).astype(int)

# Target & Encoding
target = 'Churn'
X = df.drop(target, axis=1)
y = df[target].apply(lambda x: 1 if x == 'Yes' else 0)

for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Save files
joblib.dump(model, 'xgb_churn_model.pkl')
X_test.to_csv('sample_customers.csv', index=False)
print("Training complete! Files saved.")
