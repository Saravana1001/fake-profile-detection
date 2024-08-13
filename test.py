import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'C:/Users/ADMIN/Downloads/final-v1.csv'

# Read the CSV file
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

X = data.drop('is_fake', axis=1)
y = data['is_fake']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost model
model = xgb.XGBClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Save the model to a file using joblib
joblib_file = 'C:/Users/ADMIN/Documents/xgb_model.pkl'
joblib.dump(model, joblib_file)