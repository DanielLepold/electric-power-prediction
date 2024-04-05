import keras
import tensorflow as tf
import numpy as np
import logging
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras import Sequential, Input
from .base import calculate_errors

def relu(x):
  """
  Compute the Rectified Linear Unit (ReLU) activation function element-wise.

  Parameters:
  x (numpy.ndarray): Input array.

  Returns:
  numpy.ndarray: Output array after applying the ReLU function element-wise.
  """
  return np.maximum(0, x)
def sigmoid(x):
  """
  Compute the sigmoid activation function element-wise.

  Parameters:
  x (numpy.ndarray): Input array.

  Returns:
  numpy.ndarray: Output array after applying the sigmoid function element-wise.
  """
  return 1 / (1 + np.exp(-x))
def plot_loss(history,save_path):
  plt.figure(figsize=(10, 6))
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  plt.savefig(save_path)  # Save the plot to a file
  plt.close()
def perform_regression(X_train,X_test, y_train,y_test, folder_path,df_train):
  print("Tensorflow regression started.")
  # Logging console log to the file.
  original_stdout = sys.stdout
  log_file_path ='output_tensorflow.log'

  if os.path.exists(log_file_path):
    os.remove(log_file_path)

  sys.stdout = open(log_file_path, 'a')
  print("\n-------------------------------------------------------------"
               "--------------------------------")
  print("Tensorflow regression started.")

  scaler = StandardScaler()
  X_train_sc = scaler.fit_transform(X_train)
  X_test_sc = scaler.fit_transform(X_test)  # Do not confuse train and test data
  y_train_sc = scaler.fit_transform(np.array(y_train).reshape(-1, 1))
  y_test_sc = scaler.fit_transform(np.array(y_test).reshape(-1, 1))

  print(f"X_train shape: \n{X_train_sc.shape}")

  # Build a Sequential model
  model = tf.keras.Sequential()
  model.add(Input(shape=(X_train_sc.shape[1],)))
  model.add(Dense(10, activation='sigmoid'))
  model.add(Dense(20, activation='relu'))
  model.add(Dense(8, activation='relu'))
  model.add(Dense(1))

  print(f"Model summary: \n")

  model.summary()

  tf.keras.utils.plot_model(model, show_shapes=True,show_layer_activations=True, to_file='tensorflow_model.png')

  optimizer = tf.keras.optimizers.Adam(0.001)

  loss = keras.losses.mean_squared_error

  model.compile(loss=loss, optimizer=optimizer)

  history = model.fit(X_train_sc, y_train_sc, epochs=100, validation_split=0.2)

  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  print(f"Hist tail: \n{hist.tail()}")

  plot_loss(history,"history_result.png")

  model.evaluate(X_test_sc, y_test_sc)

  y_predict_train = model.predict(X_train_sc)
  nn_predict = model.predict(X_test_sc)

  mae_train, mse_train, r2_train = calculate_errors(y_train_sc, y_predict_train)
  mae_test, mse_test, r2_test = calculate_errors(y_test_sc, nn_predict)

  results = {}
  results["train"] = {'MAE': mae_train, 'MSE': mse_train, 'R2': r2_train}
  results["test"] = {'MAE': mae_test, 'MSE': mse_test, 'R2': r2_test}

  df_results = pd.DataFrame(results).T
  print(f"Errors: \n{df_results}")

  plt.scatter(y_test_sc, nn_predict)
  plt.savefig('nn_prediction.png')

  # NN coding with basic funtions
  model.get_weights()  # Reading weights

  layer1_weights = model.get_weights()[0]
  layer1_bias = model.get_weights()[1]

  layer2_weights = model.get_weights()[2]
  layer2_bias = model.get_weights()[3]

  layer3_weights = model.get_weights()[4]
  layer3_bias = model.get_weights()[5]

  output_weights = model.get_weights()[6]
  output_bias = model.get_weights()[7]

  input = X_train_sc[0, :]
  input = input.reshape(1, -1)
  print(f"Input shape: \n{input.shape}")

  # First layer input/output calculation
  layer1_in = np.matmul(input, layer1_weights) + layer1_bias
  layer1_out = sigmoid(layer1_in)

  # Second layer input/output calculation
  layer2_in = np.matmul(layer1_out, layer2_weights) + layer2_bias
  layer2_out = relu(layer2_in)

  # Third layer input/output calculation
  layer3_in = np.matmul(layer2_out, layer3_weights) + layer3_bias
  layer3_out = relu(layer3_in)

  # Output layer input/output calculation
  output_in = np.matmul(layer3_out,output_weights) + output_bias
  output = output_in

  print(f"Output: \n{output}")
  [[1.24971628]]

  print("Tensorflow regression finished.")
  # Adjusting back the console output.
  sys.stdout = original_stdout

  return mae_test, mse_test, r2_test, 'Tensorflow Regression'



