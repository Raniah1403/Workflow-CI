import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset preprocessing
df = pd.read_csv('loan_approval_preprocessing.csv')

# Split data
X = df.drop(columns='loan_approved')
y = df['loan_approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set experiment
mlflow.set_experiment("CI Advance Training")

# autolog
mlflow.sklearn.autolog()

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc}")
