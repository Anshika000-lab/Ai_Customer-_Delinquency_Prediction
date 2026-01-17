import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_excel(
    r"C:\Users\Administrator\OneDrive - Shiv Nadar Foundation\Pictures\Desktop\Delinquency_prediction_dataset.xlsx"
)

# IMPORTANT: remove hidden spaces from column names
df.columns = df.columns.str.strip()

# Select features
X = df[
    [
        "Income_Imputed",
        "Credit_Score_Imputed",
        "Credit_Utilization",
        "Missed_Payments",
        "Loan_Balance_Imputed",
        "Debt_to_Income_Ratio",
        "Employment_Status_Encoded",
        "Credit_Card_Type_Encoded",
        "Account_Tenure"
    ]
]

y = df["Delinquent_Account"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Decision Tree
model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


