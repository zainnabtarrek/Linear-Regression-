import numpy as np
from sklearn.model_selection import train_test_split

# 1- Generate 1000 random numbers for X
np.random.seed(42)
X1 = np.random.randint(1000, size=(1000))
X2 = np.random.randint(1000, size=(1000))
X3 = np.random.randint(1000, size=(1000))

# Combine X1, X2, X3 into a matrix X
X = np.column_stack((X1, X2, X3))

# Calculate Y using the given equation
Y = 5 * X1 + 3 * X2 + 1.5 * X3 + 6

# Initialization of the weights
w1, w2, w3, w4 = 5, 3, 1.5, 6

# 2- Split the data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#  3- Initialize n_iter as the number of iterations
#  Start with a reasonably large value and observe how the cost decreases over iterations
#  If the cost stabilizes, stop the training early.
#  Initialize the learning rate with any random number
#  It's common to start with a small learning rate and increase it if the algorithm is converging too slowly
n_iterations, learning_rate = 1000, 0.3

# X_b is the xTrain after adding ones for the bias term
# Initialize weights
weights = np.array([w1, w2, w3, w4])

# Add a column of ones to X_train for the bias term
X_b_train = np.column_stack((X_train, np.ones_like(X_train[:, 0])))

# Gradient descent: W = w –[Learning_Rate(X transpose(xw-y))]
# Loss: L(w) = 0.5 * mean((xw-y)^2)
def gradientDescent(X, Y, weights, LR, iterations):
    cost_history = []
    theta_history = []

    for _ in range(iterations):
        predictions = np.dot(X, weights)
        errors = predictions - Y

        # Gradient Descent Update Rule
        gradient = np.dot(X.T, errors)
        weights -= LR * gradient

        # Loss Function (Mean Squared Error)
        cost = 0.5 * np.mean(errors ** 2)

        cost_history.append(cost)
        theta_history.append(weights.copy())

    return weights, cost_history, theta_history

# Run gradient descent
theta, cost_history, theta_history = gradientDescent(X_b_train, y_train, weights, learning_rate, n_iterations)

# Print the final weights
print("Final Weights:", theta)

# Implement the loss/cost function
def costFn(weights, X, Y):
    predictions = np.dot(X, weights)
    errors = predictions - Y
    cost = 0.5 * np.mean(errors ** 2)
    return cost

# Calculate the loss on the training data
training_loss = costFn(theta, X_b_train, y_train)
print("Training Loss:", training_loss)

# Add a column of ones to X_test for the bias term
X_b_test = np.column_stack((X_test, np.ones_like(X_test[:, 0])))

# Calculate predicted output on the test data
y_pred = np.dot(X_b_test, theta)

# Calculate the accuracy (R-squared)
def calculate_accuracy(actual, predicted):
    total_variance = np.sum((actual - np.mean(actual))**2)
    residual_variance = np.sum((actual - predicted)**2)
    r_squared = 1 - (residual_variance / total_variance)
    return r_squared

# Calculate accuracy
accuracy = calculate_accuracy(y_test, y_pred)
print("Accuracy (R-squared):", accuracy)

# Convergence check by comparing the last two elements of the cost history.
# If the last cost is less than or equal to the second-to-last cost, it's considered to have converged.

if len(cost_history) > 1 and cost_history[-1] <= cost_history[-2]:
    print("Converged successfully.")
else:
    print("Did not converge. Consider adjusting learning rate or iterations.")
