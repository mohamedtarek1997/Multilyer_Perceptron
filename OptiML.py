import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import time

# reading the data
df = pd.read_csv("X1,X2,Y.csv")
df['ones'] = np.ones(len(df))
x = df.drop(columns= "Y")
x = x.to_numpy()
y = df["Y"].to_numpy()

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state=2108602) 
train_data = np.hstack((x_train, np.reshape(y_train, (-1, 1))))

# Input parameters
N = int(input('Enter the number of neurons: '))
n = 2

# Initial guess of the weights and rho
np.random.seed(2108602)
W0 = np.random.random(N*(N+n-1))
rho = 1e-05

# The forward propagation
def y_hat(w, b, v, x):
    y_h = np.zeros(len(x))
    wb = np.hstack((w,np.reshape(b, (-1, 1))))
    for i in range(len(x)):
        row = x[i]
        z = np.tanh(np.dot(wb, np.reshape(row, (-1, 1))))  # tanh activation function
        output = np.dot(np.reshape(z, (1, -1)), np.reshape(v, (-1, 1)))  # MLP output Y = V.g(W.X)
        y_h[i] = output
    return y_h

# define the loss function by defining the MSE then the regularized error
def mse(y_h, y):
    return sum((y_h-y)**2)/len(y)

# the error function
def err_fn(W, train_data):
    X_train = train_data[:, 0:3]
    Y_train = train_data[:, 3]
    w = np.reshape(W[0:n*N],(N,n))
    v = np.reshape(W[n*N:n*N+N],(N,1))
    b = np.reshape(W[n*N+N:n*N+2*N],(N,1))
    y_h = y_hat(w, b, v, X_train) # predicted y (yhat)
    e_mse = mse(y_h, Y_train) # mean squared error
    regularization_term = 0.5 * rho * (np.linalg.norm(w) ** 2 + np.linalg.norm(v) ** 2 + np.linalg.norm(b) ** 2)
    return e_mse + regularization_term  # the regularized empirical error

# MLP Optimization
Optimization_method = "L-BFGS-B"
start = time.time()
res = minimize(err_fn, W0, args=(train_data), method=Optimization_method, tol=1e-5)

print('The optimizer results: ', res)
cost_time = time.time() - start
print('The cost time:  ', cost_time)

# Test results
final_W = res.x
w = np.reshape(final_W[0:n*N],(N,n))
v = np.reshape(final_W[n*N:n*N+N],(N,1))
b = np.reshape(final_W[n*N+N:n*N+2*N],(N,1))

print('Final Weights: ', final_W)

# The MSE for the test, and training data
y_pred_train = y_hat(w, b, v, x_train)
mse_train = mse(y_pred_train, y_train)
print('MSE for the training dataset: ', mse_train) 
y_pred = y_hat(w, b, v, x_test)
mse_test = mse(y_pred, y_test)
print('MSE for the test dataset', mse_test)

# Plotting the output
from matplotlib import projections
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
x1_vals = np.linspace(-2, 2, 50)
x2_vals = np.linspace(-3, 3, 50)
x1_mesh, x2_mesh = np.meshgrid(x1_vals, x2_vals)
x_mesh = np.column_stack((x1_mesh.flatten(), x2_mesh.flatten(), np.ones(len(x1_mesh.flatten()))))      
y = y_hat(w, b, v, x_mesh)
y_mesh = y.reshape(x1_mesh.shape)
ax = fig.add_subplot(111, projection='3d')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Predicted Values')
ax.plot_surface(x1_mesh, x2_mesh, y_mesh, cmap='viridis')
plt.show()


