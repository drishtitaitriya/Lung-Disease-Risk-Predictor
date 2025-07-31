import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("copd_dataset.csv")  # Replace with your actual file name

# Target column (change made here)
target_column = 'COPD GOLD'
if target_column not in df.columns:
    raise ValueError(f"'{target_column}' column not found in dataset. Found columns: {df.columns.tolist()}")

# Separate features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Encode categorical features
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target if categorical
if y.dtype == 'object':
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    with open('target_encoder.pkl', 'wb') as f:
        pickle.dump(target_encoder, f)
else:
    target_encoder = None

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('lung_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
