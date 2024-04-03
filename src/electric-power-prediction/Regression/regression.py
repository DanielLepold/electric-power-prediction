import logging
import os
from .linear_regression import perform_regression as linear_regression
from .polynomial_regression import perform_regression as polynomial_regression
from .ridge_regression import perform_regression as ridge_regression
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

  # Perform linear regression
  #linear_regression(X_train,X_test, y_train,y_test, folder_path,df_train)
  # Perform polynomial regression
  #polynomial_regression(X_train,X_test, y_train,y_test, folder_path)
  # Perform ridge regression
  #ridge_regression(X_train,X_test, y_train,y_test, folder_path)
  # Perform stepwise regression
  #stepwise_regression(X_train, X_test, y_train, y_test, folder_path)
  #gaussian_process_regression(X_train, X_test, y_train, y_test, folder_path)
  #tensorflow_regression(X_train,X_test, y_train,y_test, folder_path)
  random_forest_regression(X_train,X_test, y_train,y_test, folder_path)
  #k_neighbor_regression(X_train, X_test, y_train, y_test, folder_path)





  # Regression calculation with pycaret
  #pycaret_regression(df_train,df_test,folder_path)

  logging.info("REG - Regression model generation finished.")
