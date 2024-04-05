import logging
import pandas as pd
import os
from sklearn.model_selection import cross_val_predict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from .base import calculate_errors
import matplotlib.pyplot as plt

def perform_regression(X_train,X_test, y_train,y_test, folder,df_train):
  logging.info("\n-------------------------------------------------------------"
               "--------------------------------")
  logging.info("Gaussian Process regression started.")
  print("Gaussian Process regression started.")

  kernel = DotProduct() + WhiteKernel()
  gauss_proc = GaussianProcessRegressor(kernel=kernel,random_state = 0)

  gauss_proc.fit(X_train, y_train)

  # Perform cross-validation
  y_pred_train = cross_val_predict(gauss_proc, X_train, y_train,
                                   cv=5)
  y_pred_test = gauss_proc.predict(X_test)

  mae_train, mse_train, r2_train = calculate_errors(y_train, y_pred_train)
  mae_test, mse_test, r2_test = calculate_errors(y_test, y_pred_test)

  results = {}
  results["train"] = {'MAE': mae_train, 'MSE': mse_train, 'R2': r2_train}
  results["test"] = {'MAE': mae_test, 'MSE': mse_test, 'R2': r2_test}

  df_results = pd.DataFrame(results).T
  logging.info(f"Errors: \n{df_results}")

  plt.scatter(y_test, y_pred_test)
  plt.axline((420, 420), (500, 500), color='black', linewidth=1)
  plt.xlabel("y_test")
  plt.ylabel('y_p_gaussian_test')
  plt.savefig(os.path.join(folder, f'gaussian_reg_test_scatter_.png'))
  plt.show()

  logging.info("Gaussian Process regression finished.")

  return mae_test, mse_test, r2_test, 'Gaussian Process Regressor'
