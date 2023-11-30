import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import time 

columns = ['X1', 'X2', 'Y']
df = pd.read_csv('Dataset.csv', names= columns)
df.head()

### Reading and splitting the data
x = df.drop(columns= 'Y')
x = x.to_numpy()
y = df['Y'].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state=2108602)
train_data = np.hstack((x_train, np.reshape( y_train, (-1, 1) )))

### Input parameters
N = 12 
rho = 1e-05
n = x.shape[1] # number of features 


### Initial weights and weights matrix formation
def init_W(N,n):
    np.random.seed(2108602)
    W0 = np.random.random(N*n+N)
    c = np.reshape(W0[0: N*n], (N, n))
    v = np.reshape(W0[N*n:], (-1, 1))
    return W0, c, v


### The forward propagation
def RBF(x, c, sigma=1):
    phi = np.exp(-(np.linalg.norm(x-c)/ sigma)**2)
    return phi

def y_hat(x, c, v, sigma=1):
    phi_mtx = np.zeros((len(x), N))
    for i in range(len(x)):
        for j in range(N):
            phi_mtx[i,j] = RBF(x[i], c[j])
    y_h = np.squeeze(np.dot(phi_mtx, v))
    return y_h

### define the loss function by defining the MSE and the regularized error
def mse(y_h, y):
    return sum((y_h - y)**2)/(2*len(y))

def err_fn(W, train_data):
    x_train = train_data[:, 0:2]
    y_train = train_data[:, 2]
    c = np.reshape(W[0: N*n], (N, n))
    v = np.reshape(W[N*n:], (-1, 1))
    y_h = y_hat(x_train, c, v)
    regularization_term = 0.5 * rho * (np.linalg.norm(v) + np.linalg.norm(c))
    e_mse = mse(y_h, y_train)
    return regularization_term+e_mse

"""
# Grid Search
def grid(Ns, rhos):
    err_out = np.zeros(len(Ns)*len(rhos))
    err_test = np.zeros(len(Ns)*len(rhos)) 
    Weights = np.zeros((len(Ns)*len(rhos), Ns[-1]*n+Ns[-1]))
    rho_val = np.zeros(len(Ns)*len(rhos))
    N_val = np.zeros(len(Ns)*len(rhos))
    mse_test = np.zeros(len(Ns)*len(rhos))
    mse_train = np.zeros(len(Ns)*len(rhos))
    variance = np.zeros(len(Ns)*len(rhos))
    cost_time = np.zeros(len(Ns)*len(rhos))
    k = 0
    for N in Ns:
        for rho in rhos:
            W0, c, v = init_W(N,n)
            Optimization_method = "L-BFGS-B" 
            start = time.time()
            res = minimize(err_fn, W0, args=(train_data), method=Optimization_method, tol=1e-7)
            cost_time[k] = time.time() - start
            err_out[k] = round(res.fun, 3)
            Weights[k, 0:N*n+N] = res.x
            W = res.x
            c = np.reshape(W[0: N*n], (N, n))
            v = np.reshape(W[N*n:], (-1, 1))
            y_pred_test = y_hat(x_test, c, v,N)
            mse_test[k] = round(mse(y_pred_test, y_test), 3) 
            y_pred_train = y_hat(x_train, c, v, N)
            mse_train[k] = round(mse(y_pred_train, y_train), 3)
            variance[k] = round(mse_test[k] - mse_train[k], 3)
            rho_val[k] = rho
            N_val[k] = N 
            print(f"Centers: {N}, rho: {rho}, Regularized_Error: {err_out[k]}, MSE_Test: {mse_test[k]}, MSE_Train: {mse_train[k]}, Variance: {variance[k]}, cost_time: {cost_time[k]}") 
            k = k+1
            return N_val, rho_val, err_out, mse_test, mse_train, variance, cost_time """

# Optimizatio Function
def opt_rbf(W0):
    Optimization_method = "L-BFGS-B"
    start = time.time()
    res = minimize(err_fn, W0, args=(train_data), method=Optimization_method, tol=1e-7)
    cost_time = time.time() - start
    return cost_time, res

def W_last(res):
    final_W = res.x
    c = np.reshape(final_W[0: N*n], (N, n))
    v = np.reshape(final_W[N*n:], (-1, 1))
    return c, v
     

# Plotting the output
def out_plot(c, v):
    from matplotlib import projections
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(dpi=120)
    x1_vals = np.linspace(-2, 2, 50)
    x2_vals = np.linspace(-3, 3, 50)
    x1_mesh, x2_mesh = np.meshgrid(x1_vals, x2_vals)
    x_mesh = np.column_stack((x1_mesh.flatten(), x2_mesh.flatten()))
    y = y_hat(x_mesh, c, v)
    y_mesh = y.reshape(x1_mesh.shape)
    ax = fig.add_subplot(111, projection='3d')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Predicted Values Using RBF')
    ax.plot_surface(x1_mesh, x2_mesh, y_mesh, cmap='viridis')
    plt.show()

    