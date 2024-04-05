import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from .base import calculate_errors
from sklearn.model_selection import cross_val_predict



def perform_regression(X_train,X_test, y_train,y_test, folder,df_train):
  logging.info("\n-------------------------------------------------------------"
               "--------------------------------")
  logging.info("Stepwise regression started.")
  print("Stepwise regression started.")

  sfs = SequentialFeatureSelector(estimator=LinearRegression(),
                                  n_features_to_select='auto', scoring='r2',
                                  direction='backward')
  sfs.fit(X_train, y_train)

  X_train_selected = sfs.transform(X_train)
  X_test_selected = sfs.transform(X_test)

  reg = LinearRegression()
  reg.fit(X_train_selected, y_train)

  y_p_train = cross_val_predict(reg, X_train_selected, y_train, cv=5)
  y_p_test = reg.predict(X_test_selected)

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
  plt.savefig(os.path.join(folder, f'stepwise_reg_scatter.png'))
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
  plt.savefig(os.path.join(folder, f'stepwise_reg_qq_plot.png'))
  plt.show()

  # Uncertanity of prediction
  X2 = sm.add_constant(X_train_selected)
  model = sm.OLS(y_train, X2).fit()
  logging.info(f"Summary of Regression in Statsmodels: \n{model.summary()}")

  # extract p-values for all predictor variables
  # for x in range(0, 3):
  #   logging.info(model.pvalues[x])

  logging.info("Stepwise regression finished.")

  # logging.info(f"X column support: \n{X_train.columns[sfs.get_support()]}")
  return mae_test, mse_test, r2_test, 'Sequential Feature Selector'

