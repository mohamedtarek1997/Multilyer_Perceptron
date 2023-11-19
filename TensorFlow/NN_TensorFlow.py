import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# reading the data
df = pd.read_csv('X1,X2,Y.csv')
x = df.drop(columns= "Y")
x = x.to_numpy()
y = df["Y"].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state= 2057785)


def plot_history(history):
  fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))
  ax1.plot(history.history['loss'])
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('MSE')
  ax1.set_title('Epochs VS MSE')
  ax1.grid(True)
  plt.show()

def train_model(x_train, y_train, epochs, lr, dropout_prob, N, batch_size):

    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(N, activation= "tanh", input_shape= (2,)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation= None)
    ])

    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), metrics=['mean_squared_error'], loss= 'mean_squared_error')

    history = nn_model.fit(x_train, y_train, batch_size= batch_size, epochs= epochs, verbose=0)
    
    return nn_model, history


model, history = train_model(x_train, y_train, 1500, 0.001, 0.0, 5, 10)

plot_history(history)
y_hat_test = np.squeeze(model.predict(x_test))
mse_test = sum((y_hat_test - y_test)**2)/len(y_test)
mse_test


# Plotting the output
from matplotlib import projections
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(dpi=120)
x1_vals = np.linspace(-2, 2, 50)
x2_vals = np.linspace(-3, 3, 50)
x1_mesh, x2_mesh = np.meshgrid(x1_vals, x2_vals)
x_mesh = np.column_stack((x1_mesh.flatten(), x2_mesh.flatten()))      
y = model.predict(x_mesh)
y_mesh = y.reshape(x1_mesh.shape)
ax = fig.add_subplot(111, projection='3d')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Predicted Values Using TensorFlow')
ax.plot_surface(x1_mesh, x2_mesh, y_mesh, cmap='viridis')
plt.show()