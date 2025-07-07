import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your CSV from Downloads
file_path = "sensor_data.csv"
data = pd.read_csv(file_path)

# Define features and target
X = data[['temperature', 'vibration', 'pressure']]
y = data['failure']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print results
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Sample predictions
print("\n📦 Sample Predictions:")
sample_inputs = X_test.head()
sample_outputs = model.predict(sample_inputs)
for i, row in sample_inputs.iterrows():
    print(f"Input: {row.to_dict()} → Predicted Failure: {sample_outputs[list(sample_inputs.index).index(i)]}")
