import logging
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from .base import calculate_errors
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_predict


def perform_regression(X_train,X_test, y_train,y_test, folder,df_train):
  logging.info("\n-------------------------------------------------------------"
               "--------------------------------")
  logging.info("Lasso regression started.")
  print("Lasso regression started.")

  # It scales the values around 0
  scaler = StandardScaler()
  X_poly_train = scaler.fit_transform(X_train)
  X_poly_test = scaler.transform(X_test)
  y_poly_train = y_train
  y_poly_test = y_test

  np.mean(X_poly_train, axis=0)

  # We have max R2 value in 0
  # After alpha = 16 the params tend to have 0 value, therefore zero more influence
  # We have 4 parameters (columns, which can be seen also in the diagrams)
  reg = Lasso(alpha=0)
  logging.info(f"Lasso reg: \n{reg}")

  reg.fit(X_poly_train, y_poly_train)
  # Perform cross-validation
  y_pred_train = cross_val_predict(reg, X_poly_train, y_poly_train,
                                         cv=5)
  y_pred_test = reg.predict(X_poly_test)

  mae_train, mse_train, r2_train = calculate_errors(y_train, y_pred_train)
  mae_test, mse_test, r2_test = calculate_errors(y_test, y_pred_test)

  results = {}
  results["train"] = {'MAE': mae_train, 'MSE': mse_train, 'R2': r2_train}
  results["test"] = {'MAE': mae_test, 'MSE': mse_test, 'R2': r2_test}

  df_results = pd.DataFrame(results).T
  logging.info(f"Errors: \n{df_results}")

  logging.info(
    f"Estimated coefficients for regression: \n{reg.coef_}")
  logging.info(f"Intercept: \n{reg.intercept_}")

  plt.scatter(y_poly_train, y_pred_train)
  plt.axline((420, 420), (500, 500), color='black', linewidth=1)
  plt.xlabel("y_train")
  plt.ylabel('y_p_lasso_train')
  plt.savefig(os.path.join(folder, f'lasso_reg_train_scatter_.png'))
  plt.show()

  plt.scatter(y_poly_test, y_pred_test)
  plt.axline((420, 420), (500, 500), color='black', linewidth=1)
  plt.xlabel("y_test")
  plt.ylabel('y_p_lasso_test')
  plt.savefig(os.path.join(folder, f'lasso_reg_test_scatter_.png'))
  plt.show()

  alpha_array = np.linspace(0, 30, 100)
  r2_train = []
  r2_test = []
  params = []


  for alpha in alpha_array:
    clf = Lasso(alpha=alpha)
    clf.fit(X_poly_train, y_poly_train)

    lasso_train_predict = cross_val_predict(clf, X_poly_train, y_poly_train, cv=5)
    lasso_test_predict = clf.predict(X_poly_test)

    r2_train_temp = r2_score(y_poly_train, lasso_train_predict)
    r2_test_temp = r2_score(y_poly_test, lasso_test_predict)

    r2_train.append(r2_train_temp)
    r2_test.append(r2_test_temp)
    params.append(clf.coef_)

  plt.plot(alpha_array, r2_train, alpha_array, r2_test)
  plt.legend(['train', 'test'])
  plt.savefig(os.path.join(folder, f'lasso_alpha_.png'))
  plt.show()

  # Az a parameterek a jok, amik nem 0-hoz tartanak.
  plt.plot(alpha_array, params)
  plt.savefig(os.path.join(folder, f'lasso_alpha_params.png'))
  plt.show()


  logging.info("Lasso regression finished.")

  return mae_test, mse_test, r2_test, 'Lasso Regression'
