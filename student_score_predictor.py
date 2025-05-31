
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("student_scores.csv")

# Visualize the data
plt.scatter(df['Hours'], df['Scores'])
plt.title("Study Hours vs Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.grid(True)
plt.show()

# Prepare data
X = df[['Hours']]
y = df['Scores']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
for i in range(len(X_test)):
    print(f"Hours Studied: {X_test.iloc[i].values[0]}, Actual Score: {y_test.iloc[i]}, Predicted Score: {y_pred[i]:.2f}")

# Accuracy
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title("Regression Line - Hours vs Score")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.grid(True)
plt.show()
