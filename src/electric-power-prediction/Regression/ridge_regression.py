import logging
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from .base import calculate_errors
from sklearn.model_selection import cross_val_predict


def perform_regression(X_train,X_test, y_train,y_test, folder,df_train):
  logging.info("\n-------------------------------------------------------------"
               "--------------------------------")
  logging.info("Ridge regression started.")
  print("Ridge regression started.")

  # It scales the values around 0
  scaler = StandardScaler()
  X_poly_train = scaler.fit_transform(X_train)
  X_poly_test = scaler.transform(X_test)
  y_poly_train = y_train
  y_poly_test = y_test

  np.mean(X_poly_train, axis=0)

  # If the alpha is 0, it means it is a LinearRegression model only
  # We have max R2 close to zero
  ridge_reg = Ridge(alpha=0.1)
  logging.info(f"Ridge reg: \n{ridge_reg}")

  ridge_reg.fit(X_poly_train, y_poly_train)
  # Perform cross-validation
  y_ridge_pred_train = cross_val_predict(ridge_reg, X_poly_train, y_poly_train, cv=5)
  y_ridge_pred_test = ridge_reg.predict(X_poly_test)

  mae_train, mse_train, r2_train = calculate_errors(y_train, y_ridge_pred_train)
  mae_test, mse_test, r2_test = calculate_errors(y_test, y_ridge_pred_test)

  results = {}
  results["train"] = {'MAE': mae_train, 'MSE': mse_train, 'R2': r2_train}
  results["test"] = {'MAE': mae_test, 'MSE': mse_test, 'R2': r2_test}

  df_results = pd.DataFrame(results).T
  logging.info(f"Errors: \n{df_results}")

  logging.info(f"Estimated coefficients for regression: \n{ridge_reg.coef_}")
  logging.info(f"Intercept: \n{ridge_reg.intercept_}")

  plt.scatter(y_poly_train, y_ridge_pred_train)
  plt.axline((420, 420), (500, 500), color='black', linewidth=1)
  plt.xlabel("y_train")
  plt.ylabel('y_p_ridge_train')
  plt.savefig(os.path.join(folder, f'ridge_reg_train_scatter_.png'))
  plt.show()

  plt.scatter(y_poly_test, y_ridge_pred_test)
  plt.axline((420, 420), (500, 500), color='black', linewidth=1)
  plt.xlabel("y_test")
  plt.ylabel('y_p_ridge_test')
  plt.savefig(os.path.join(folder, f'ridge_reg_test_scatter_.png'))
  plt.show()

  alpha_array = np.linspace(0, 1000, 10000)

  r2_train = []
  r2_test = []
  params = []

  for alpha in alpha_array:
    clf = Ridge(alpha=alpha)
    clf.fit(X_poly_train, y_poly_train)

    ridge_train_predict = cross_val_predict(clf, X_poly_train, y_poly_train, cv=5)
    ridge_test_predict = clf.predict(X_poly_test)

    r2_train_temp = r2_score(y_poly_train, ridge_train_predict)
    r2_test_temp = r2_score(y_poly_test, ridge_test_predict)

    r2_train.append(r2_train_temp)
    r2_test.append(r2_test_temp)
    params.append(clf.coef_)

  plt.plot(alpha_array, r2_train, alpha_array, r2_test)
  plt.legend(['train', 'test'])
  plt.savefig(os.path.join(folder, f'ridge_alpha_.png'))
  plt.show()

  noise = np.random.normal(0, 1000, len(y_poly_train))
  y_poly_train_scaled_noisy = y_poly_train + noise
  noise = np.random.normal(0, 1000, len(y_poly_test))
  y_poly_test_scaled_noisy = y_poly_test + noise

  plt.scatter(y_poly_train, y_poly_train_scaled_noisy)
  plt.savefig(os.path.join(folder, f'ridge_scatter_noisy_.png'))
  plt.show()

  logging.info("Ridge regression finished.")

  return mae_test, mse_test, r2_test, 'Ridge Regression'

