import os
import sys
import logging
import pandas as pd
sys.path.insert(0, '/Regression/')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def plot_linear_regression(reg, X_train,y_train,y_pred_train,folder,factor):
  # Get the coefficients of the linear regression model
  # y = b0 + b1 x
  # slope = b0
  # intercept = b1
  slope = reg.coef_[0]
  intercept = reg.intercept_
  # Plot the scatter diagram
  plt.scatter(X_train, y_train, color='blue', label='Data')

  # Plot the linear regression line
  plt.plot(X_train, y_pred_train, color='red',
           label='Linear Regression')

  # Add equation to the plot
  plt.text(25, 500, f'y = {slope:.4f}x + {intercept:.2f}', fontsize=12)

  # Add labels and legend
  plt.xlabel(factor)
  plt.ylabel('PE')
  plt.legend()

  # Save the plot
  plt.savefig(os.path.join(folder, f'linear_reg_{factor}_PE.png'))

  # Show plot
  plt.show()

def calculate_errors(y_true,y_pred):
  mae = mean_absolute_error(y_true, y_pred)
  mse = mean_squared_error(y_true, y_pred)
  r2 = r2_score(y_true, y_pred)

  return mae, mse, r2
def create_model(df_train,df_test,output_folder_path):
  print("Linear regression started.")
  results = {}
  for factor in ['AT', 'AP', 'RH', 'V'] :
    X_train = np.array(df_train[factor])
    # PE train
    y_train = np.array(df_train.PE)

    # Calculating linear regression for [factor] and PE
    logging.info("------------------------------------------------------------")
    logging.info(f"Calculating linear regression for {factor} and PE.")
    logging.info(f"{factor}factor, X_train: \n{X_train}")
    logging.info(f"PE factor, y_train: \n{y_train}")

    # Reshaping the data into 2D from 1D
    X_train_reshaped = X_train.reshape(-1, 1)

    # Create and fit the linear regression model
    reg = LinearRegression()
    reg.fit(X_train_reshaped, y_train)
    y_pred_train = reg.predict(X_train_reshaped)

    plot_linear_regression(reg, X_train, y_train, y_pred_train,
                           output_folder_path,factor)

    # Checking the errors:
    X_test = np.array(df_test[factor])
    y_test = np.array(df_test["PE"])

    # Reshape the test data
    X_test_reshaped = X_test.reshape(-1, 1)

    # Predict using the trained model on test data
    y_pred_test = reg.predict(X_test_reshaped)

    # Calculating errors
    mae_test, mse_test, r2_test = calculate_errors(y_test,y_pred_test)
    mae_train, mse_train, r2_train = calculate_errors(y_train, y_pred_train)

    logging.info(
      f"Mean absolute errors, test vs. train: \n{mae_test, mae_train}")
    logging.info(
      f"Mean squared errors, test vs. train: \n{mse_test, mse_train}")
    logging.info(f"R2 score, test vs. train: \n{r2_test, r2_train}")

    results[factor] = {'MAE': mae_test, 'MSE': mse_test, 'R2': r2_test}

  df_results = pd.DataFrame(results).T
  logging.info(f"Errors: \n{df_results}")
  print("Linear regression finished.")










