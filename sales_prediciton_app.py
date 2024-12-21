import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

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
        self.losses = []
        for epoch in range(epochs):
            self.forward(X)
            loss = np.mean((self.Z2 - y.reshape(-1, 1)) ** 2)
            self.losses.append(loss)
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
    st.title("Anthony's Enhanced Sales Prediction App")
    st.write("Predict sales, analyze trends, and visualize advertising data.")

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Data Analytics", "Model Performance"])

    if page == "Prediction":
        st.header("Sales Prediction")
        st.write("Use the sliders to input advertising budgets.")

        # Input sliders
        tv_budget = st.slider("TV", 0, int(data['TV'].max()), step=1)
        radio_budget = st.slider("Radio", 0, int(data['Radio'].max()), step=1)
        newspaper_budget = st.slider("Newspaper", 0, int(data['Newspaper'].max()), step=1)

        if st.button("Estimate"):
            # Normalize inputs
            user_input = np.array([[tv_budget, radio_budget, newspaper_budget]])
            user_input = (user_input - X.mean(axis=0)) / X.std(axis=0)

            # Predict using the trained model
            prediction = nn.forward(user_input)
            st.success(f"Estimated Sales: {prediction[0][0]:.2f}")

    elif page == "Data Analytics":
        st.header("Data Analytics")
        st.write("Explore the dataset and its statistics.")
        st.write("**Summary Statistics:**")
        st.write(data.describe())

        st.write("**Feature Correlation with Sales:**")
        fig, ax = plt.subplots()
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.write("**Scatterplot of Features vs. Sales:**")
        for feature in ['TV', 'Radio', 'Newspaper']:
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[feature], y=data['Sales'], ax=ax)
            st.pyplot(fig)

    elif page == "Model Performance":
        st.header("Model Performance")
        st.write("**Loss Over Epochs:**")
        fig, ax = plt.subplots()
        ax.plot(range(len(nn.losses)), nn.losses)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Over Epochs")
        st.pyplot(fig)

        # Model Testing
        y_pred = nn.forward(X_test).flatten()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

if __name__ == '__main__':
    main()
