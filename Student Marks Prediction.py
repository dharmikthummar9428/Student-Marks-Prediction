import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [10, 20, 35, 40, 50, 60, 65, 75, 85, 95]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Marks"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression 
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print predictions
print("Actual Marks:", y_test.values)
print("Predicted Marks:", y_pred)

# Predict for new data
hours = pd.DataFrame([[7.5]], columns=["Hours"])
predicted_marks = model.predict(hours)

print("Predicted marks for 7.5 study hours:", predicted_marks[0])

# Visualization
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.legend()
plt.show()
