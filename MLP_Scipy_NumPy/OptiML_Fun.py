import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import time

# reading the data
columns = ["X1", "X2", "Y"]
df = pd.read_csv("Dataset.csv", names = columns)
df['ones'] = np.ones(len(df))
x = df.drop(columns= "Y")
x = x.to_numpy()
y = df["Y"].to_numpy()

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state=2108602) 
train_data = np.hstack((x_train, np.reshape(y_train, (-1, 1))))  

# Input parameters
N = 11  # number of neurons
n = 2   # number of features
rho = 0.01

# Initial guess of the weights and rho
def W_I():
    np.random.seed(2108602)
    W0 = np.random.random(N*n+N+N) 
    w = np.reshape(W0[0:n*N],(N,n))
    v = np.reshape(W0[n*N:n*N+N],(N,1))
    b = np.reshape(W0[n*N+N:n*N+2*N],(N,1)) 
    return W0, w, b, v 

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

# the regularized error function 
def err_fn(W, train_data):
    X_train = train_data[:, 0:3] 
    Y_train = train_data[:, 3]
    w = np.reshape(W[0:n*N],(N,n))
    v = np.reshape(W[n*N:n*N+N],(N,1))
    b = np.reshape(W[n*N+N:n*N+2*N],(N,1))
    y_h = y_hat(w, b, v, X_train) # predicted y (yhat)
    e_mse = mse(y_h, Y_train) # mean squared error
    regularization_term = 0.5 * rho * (np.linalg.norm(w) + np.linalg.norm(v) + np.linalg.norm(b))
    return e_mse + regularization_term  # the regularized empirical error

# MLP Optimization
def opt_mlp(W0):
    Optimization_method = "L-BFGS-B"
    start = time.time()
    res = minimize(err_fn, W0, args=(train_data), method=Optimization_method, tol=1e-7)
    cost_time = time.time() - start
    return cost_time, res

# Test results
def W_last(res):    
    final_W = res.x
    w = np.reshape(final_W[0:n*N],(N,n))
    v = np.reshape(final_W[n*N:n*N+N],(N,1))
    b = np.reshape(final_W[n*N+N:n*N+2*N],(N,1))
    return w, b, v




# Plotting the output
def out_plot(w, b, v):
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
    plt.title('Predicted Values Using MLP')
    ax.plot_surface(x1_mesh, x2_mesh, y_mesh, cmap='viridis')
    plt.show()


