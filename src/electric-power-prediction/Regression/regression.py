import logging
import os
import pandas as pd
from .linear_regression import perform_regression as linear_regression
from .linear_regression import perform_simple_linear_regression as simple_linear_regression

from .polynomial_regression import perform_regression as polynomial_regression
from .ridge_regression import perform_regression as ridge_regression
from .lasso_regression import perform_regression as lasso_regression
from .stepwise_regression import perform_regression as stepwise_regression
from .gaussian_process_regressor import perform_regression as gaussian_process_regression
from .regression_pycaret import perform_regression as pycaret_regression
from .tensorflow_regression import perform_regression as tensorflow_regression
from .random_forest_regression import perform_regression as random_forest_regression
from .k_neighbors_regression import perform_regression as k_neighbor_regression


def create_models(df_train, df_test,folder):
  logging.info("REG - Regression model generation started.")

  folder_path = folder + "/Regression"
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

  X_train = df_train.iloc[:,:-1].copy()
  y_train = df_train["PE"].copy()
  X_test = df_test.iloc[:, :-1].copy()
  y_test = df_test["PE"].copy()

  results = simple_linear_regression(df_train,df_test,  folder_path)
  regression_functions = [
    linear_regression,
    polynomial_regression,
    lasso_regression,
    stepwise_regression,
    gaussian_process_regression,
    tensorflow_regression,
    random_forest_regression,
    k_neighbor_regression
  ]

  for regression_function in regression_functions:
    mae_test, mse_test, r2_test, model = regression_function(X_train, X_test,
                                                             y_train, y_test,
                                                             folder_path,df_train)
    results[model] = {'MAE': mae_test, 'MSE': mse_test, 'R2': r2_test}

  df_results = pd.DataFrame(results).T
  df_results_sorted = df_results.sort_values(by='R2', ascending=False)

  logging.info(f"Errors: \n{df_results_sorted}")

  # Regression calculation with pycaret
  #pycaret_regression(df_train,df_test,folder_path)

  logging.info("REG - Regression model generation finished.")
