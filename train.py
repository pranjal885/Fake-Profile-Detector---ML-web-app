import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("dataset/fake_social_media.csv")

# -------------------------------
# 2. Encode Categorical Column
# -------------------------------
df['platform'] = df['platform'].map({
    'Twitter': 0,
    'Instagram': 1,
    'Facebook': 2
})

# -------------------------------
# 3. Separate Features & Target
# -------------------------------
X = df.drop('is_fake', axis=1)
y = df['is_fake']

# -------------------------------
# 4. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------
# 6. Train Model
# -------------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# -------------------------------
# 7. Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))


# -------------------------------
# 8. Save Model & Scaler
# -------------------------------
pickle.dump(model, open("fake_profile_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model and scaler saved successfully")
