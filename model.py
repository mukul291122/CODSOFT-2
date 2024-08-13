import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# Load the dataset
data_file = r'C:\Mukul\Customer\dataset\Churn_Modelling.csv'
df = pd.read_csv(data_file)

# Drop columns that are not needed
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode categorical variables
label_encoder = LabelEncoder()
df['Geography'] = label_encoder.fit_transform(df['Geography'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Split the data into features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)
logistic_pred = logistic_model.predict(X_test)
logistic_acc = accuracy_score(y_test, logistic_pred)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)

# Function to evaluate model performance
def evaluate_model(model_name, y_test, y_pred):
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Evaluate Logistic Regression
evaluate_model("Logistic Regression", y_test, logistic_pred)

# Evaluate Random Forest
evaluate_model("Random Forest", y_test, rf_pred)

# Evaluate Gradient Boosting
evaluate_model("Gradient Boosting", y_test, gb_pred)

# Save the scaler
model_dir = r'C:\Mukul\Customer'
scaler_file = os.path.join(model_dir, 'scaler.joblib')
joblib.dump(scaler, scaler_file)
print(f"Scaler saved to {scaler_file}")

# Determine the best model and save it
best_model_name = None
best_model = None
best_acc = 0

if logistic_acc > best_acc:
    best_model_name = "Logistic Regression"
    best_model = logistic_model
    best_acc = logistic_acc

if rf_acc > best_acc:
    best_model_name = "Random Forest"
    best_model = rf_model
    best_acc = rf_acc

if gb_acc > best_acc:
    best_model_name = "Gradient Boosting"
    best_model = gb_model
    best_acc = gb_acc

print(f"The best model is {best_model_name} with an accuracy of {best_acc}")

# Save the best model to a file
model_file = os.path.join(model_dir, f'{best_model_name}_model.joblib')
joblib.dump(best_model, model_file)
print(f"Best model saved to {model_file}")
