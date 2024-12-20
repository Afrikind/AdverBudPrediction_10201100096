import streamlit as st
import pandas as pd
import numpy as np

# Neural Network Implementation
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return x > 0

    def forward(self, X):
        # Forward pass
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2

    def backward(self, X, y, learning_rate):
        # Backward pass
        m = X.shape[0]
        dZ2 = self.Z2 - y.reshape(-1, 1)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

# Load and preprocess dataset
data = pd.read_csv('advertising.csv')
X = data[['TV', 'Radio', 'Newspaper']].values
y = data['Sales'].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Split data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train Neural Network
nn = SimpleNeuralNetwork(input_size=3, hidden_size=5, output_size=1)
nn.train(X_train, y_train, epochs=1000, learning_rate=0.01)

# Streamlit Web Application
def main():
    st.title("Anthony's 'Sales Prediction App")
    st.write("You can Predict sales based on advertising budgets."
    
        " Please Enter ")

    # User input fields with validation and help text
    tv_budget = st.number_input("TV", min_value=0.0, step=0.1, help="Enter TV advertisement budget in dollars")
    radio_budget = st.number_input("Radio", min_value=0.0, step=0.1, help="Enter Radio advertisement budget in dollars")
    newspaper_budget = st.number_input("Newspaper", min_value=0.0, step=0.1, help="Enter Newspaper advertisement budget in dollars")

    if st.button("Estimate"):
        # Validate inputs
        if tv_budget == 0.0 and radio_budget == 0.0 and newspaper_budget == 0.0:
            st.error("Please enter valid values for all ad budgets.")
        else:
            # Normalize inputs
            user_input = np.array([[tv_budget, radio_budget, newspaper_budget]])
            user_input = (user_input - X.mean(axis=0)) / X.std(axis=0)

            # Predict using the trained model
            prediction = nn.forward(user_input)
            st.success(f"Estimate: {prediction[0][0]:.2f}")

if __name__ == '__main__':
    main()
