from Opt_RBF_Fun import *


# Initial Guess
W0, c, v = init_W(N,n)

# Initial MSE
y_init_train = y_hat(x_train, c, v)
mse_init_train = mse(y_init_train, y_train)
y_init_test = y_hat(x_test, c, v)
mse_init_test = mse(y_init_test, y_test)

# Running the Optimizer
cost_time, res = opt_rbf(W0) 

# Final Weights
c, v = W_last(res)

# The MSE for the test, and training data
y_pred_train = y_hat(x_train, c, v)
mse_train = mse(y_pred_train, y_train)
y_pred = y_hat(x_test, c, v)
mse_test = mse(y_pred, y_test)

# printing the outputs
print(f"The number of neurons: {N}")
print("Sigma = 1")
print("Rho : ", rho)
print('Optimization Solver: L-BFGS-B')
print('Number of function evaluations: ', res.nfev)
print('Number of gradient evaluations: ', res.njev)
print('Time to optimize the network:  ', cost_time)
print('Training Error:  ', mse_train)
print('Testing Error:  ', mse_test)
print('Initial Training Error:  ', mse_init_train)
print('Initial Testing Error:  ', mse_init_test)

# 3_D Plot
out_plot(c, v)