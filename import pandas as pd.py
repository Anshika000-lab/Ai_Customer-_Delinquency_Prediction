import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset (use raw string for Windows path)
df = pd.read_excel(
    r"C:\Users\Administrator\OneDrive - Shiv Nadar Foundation\Pictures\Desktop\Delinquency_prediction_dataset.xlsx"
)

# Quick check
df.columns = df.columns.str.strip()

print(df.columns)

# Define features (X) and target (y)
X = df[['Income_Imputed',
        'Credit_Utilization',
        'Loan_Balance_Imputed',
        'Credit_Score_Imputed',
        'Missed_Payments',
        'Debt_to_Income_Ratio',
        'Account_Tenure',
        'Employment_Status_Encoded',
        'Credit_Card_Type_Encoded']]

y = df['Delinquent_Account']

# Train-test split (keeps 16% delinquency ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Logistic Regression with class balancing
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Coefficient interpretation
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\nModel Coefficients:")
print(coefficients)
# Remove unnamed columns created by Excelfrom sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)
# Predict probability of delinquency (class = 1)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Set a lower threshold to improve recall
threshold = 0.25
y_pred = (y_prob >= threshold).astype(int)


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))





