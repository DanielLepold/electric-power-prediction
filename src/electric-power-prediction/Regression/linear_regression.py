import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from .base import calculate_errors

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



def perform_regression(X_train,X_test, y_train,y_test, folder,df_train):
  logging.info("\n-------------------------------------------------------------"
               "--------------------------------")
  logging.info("Linear regression started.")

  reg = LinearRegression()
  reg.fit(X_train, y_train)

  # Estimated coeffictions for linear regression
  logging.info(f"Estimated coefficients for linear regression: \n{reg.coef_}")
  logging.info(f"Intercept: \n{reg.intercept_}")

  logging.info(f"Reg coefficients: \n{pd.DataFrame(reg.coef_, df_train.columns[:-1], columns=['reg_coef'])}")

  y_p_train = reg.predict(X_train)
  y_p_test = reg.predict(X_test)

  # Calculating errors
  mae_train, mse_train, r2_train = calculate_errors(y_train, y_p_train)
  mae_test, mse_test, r2_test = calculate_errors(y_test, y_p_test)

  results = {}
  results["train"] = {'MAE': mae_train, 'MSE': mse_train, 'R2': r2_train}
  results["test"] = {'MAE': mae_test, 'MSE': mse_test, 'R2': r2_test}

  df_results = pd.DataFrame(results).T
  logging.info(f"Errors: \n{df_results}")

  # Scatter plot for test vs prediction
  plt.scatter(y_test, y_p_test)
  plt.axline((420, 420), (500, 500), color='black', linewidth=1)
  plt.xlabel("y_test")
  plt.ylabel('y_p_test')
  plt.savefig(os.path.join(folder, f'linear_reg_scatter.png'))
  plt.show()

  # Determine adjusted square error
  N = X_test.shape[0]
  K = X_test.shape[1]
  logging.info(f"N: \n{N}")
  logging.info(f"K: \n{K}")
  adj_r2 = 1 - (1 - r2_test) * ((N - 1) / (N - K))
  logging.info(f"Adjusted r2: \n{adj_r2}")

  error = y_p_train - y_train
  error_std = (error - np.mean(error)) / np.std(error)

  # QQ plot for train vs preidiction
  sm.qqplot(error_std, line='45')
  plt.savefig(os.path.join(folder, f'linear_reg_qq_plot.png'))
  plt.show()

  # Uncertanity of prediction
  X2 = sm.add_constant(X_train)
  model = sm.OLS(y_train, X2).fit()
  logging.info(f"Summary of Regression in Statsmodels: \n{model.summary()}")

  # extract p-values for all predictor variables
  # for x in range(0, 3):
  #   logging.info(model.pvalues[x])

  logging.info("Linear regression finished.")

def perform_simple_linear_regression(df_train,df_test,output_folder_path):
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










