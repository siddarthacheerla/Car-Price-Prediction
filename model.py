import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import json

print("Current working directory:", os.getcwd())
csv_path = 'car_data.csv'
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV file not found at path: {csv_path}")

# Load dataset
df = pd.read_csv(csv_path)
print("Columns in CSV:", df.columns.tolist())

# Strip whitespace from column names if any
df.columns = df.columns.str.strip()

# Drop irrelevant columns like 'car_ID' and 'CarName' (unique identifiers or non-numeric)
df = df.drop(['car_ID', 'CarName'], axis=1)

# Select features and target 'price'
features = df.drop('price', axis=1)
target = df['price']

# Convert categorical columns to dummies
features = pd.get_dummies(features, drop_first=True)

# Save the columns used for training (feature names)
feature_columns = features.columns.tolist()
with open('feature_columns.json', 'w') as f:
    json.dump(feature_columns, f)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'car_price_model.pkl')
print("Model trained and saved as car_price_model.pkl")

# Model evaluation
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.2f}")
print(f"Test R²: {r2:.2f}")
