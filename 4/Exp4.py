
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import tkinter as tk
from tkinter import messagebox

# Read the dataset
data = pd.read_csv('titanic_dataset.csv')  # Replace with the actual file path

# Find the number of columns in the dataset
num_columns = len(data.columns)
print("Number of columns:", num_columns)

# Find the number of passengers in the dataset
num_passengers = len(data)
print("Number of passengers:", num_passengers)

# Find the age of a passenger who survived
age_of_survivor = data[data['Survived'] == 1]['Age'].dropna()
print("Age of a passenger who survived:", age_of_survivor.values[0])

# Find the complete info of the dataset
data_info = data.info()
print(data_info)

# Check for null data in the dataset
has_null = data.isnull().any().any()
print("Does the dataset have null data:", has_null)

# Count the null values in the dataset
null_counts = data.isnull().sum()
print("Null counts:\n", null_counts)

# Remove the null values from the dataset
data_cleaned = data.dropna()
# Create dummy variables if applicable
data_cleaned = pd.get_dummies(data_cleaned, columns=['Sex'], drop_first=True)

# Splitting the data for model training
X = data_cleaned.drop('Survived', axis=1)
y = data_cleaned['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)
