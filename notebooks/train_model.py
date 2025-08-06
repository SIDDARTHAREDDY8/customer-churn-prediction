import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv("data/telco_churn.csv")

# Drop rows with missing values (especially TotalCharges)
data = data.dropna()

# Drop 'customerID' and 'Churn' from features
X = data.drop(['customerID', 'Churn'], axis=1)
y = data['Churn']

# Encode categorical features
label_encoder = LabelEncoder()
for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = label_encoder.fit_transform(X[column])

# Encode target variable
y = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")

