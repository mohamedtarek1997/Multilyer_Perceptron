from OptiML_Fun import *

# Initial guess
W0, w, b, v = W_I()

# Initial MSE
y_init_train = y_hat(w, b, v, x_train)
mse_init_train = mse(y_init_train, y_train)
y_init_test = y_hat(w, b, v, x_test)
mse_init_test = mse(y_init_test, y_test)

# Running the Optimizer
cost_time, res = opt_mlp(W0)

# Test results
w, b, v = W_last(res) 


# The MSE for the test, validation, and training data 
y_pred_train = y_hat(w, b, v, x_train)
mse_train = mse(y_pred_train, y_train)
y_pred = y_hat(w, b, v, x_test)
mse_test = mse(y_pred, y_test)

# Plotting the output
out_plot(w, b, v)

# printing the outputs
print(f"The number of neurons: {N}")
print("Sigma = 1")
print("Rho = 0.01")
print('Optimization Solver: L-BFGS-B')
print('Number of function evaluations: ', res.nfev)
print('Number of gradient evaluations: ', res.njev)
print('Time to optimize the network:  ', cost_time)
print('Training Error:  ', mse_train)
print('Testing Error:  ', mse_test)
print('Initial Training Error:  ', mse_init_train)
print('Initial Testing Error:  ', mse_init_test)



