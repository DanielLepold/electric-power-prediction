from sklearn.preprocessing import PolynomialFeatures
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .base import calculate_errors

def perform_regression(X_train,X_test, y_train,y_test, folder):
  logging.info("\n-------------------------------------------------------------"
               "--------------------------------")
  logging.info("Polynomial regression started.")

  poly_reg = PolynomialFeatures(degree=2, interaction_only=False)

  # Adjusting poly data

  # The fit_transform method should only be called on the training data
  # (X_train), not on the test data as well. Test data should only be
  # transformed based on the pre-determined polynomial features.
  X_poly_train = poly_reg.fit_transform(X_train)
  X_poly_test= poly_reg.transform(X_test)
  y_poly_train = y_train
  y_poly_test = y_test

  poly_reg.get_feature_names_out()
  logging.info(f"Feature names: \n{poly_reg.get_feature_names_out()}")

  reg_pol = LinearRegression()
  reg_pol.fit(X_poly_train, y_poly_train)

  # Estimated coeffictions for linear regression
  logging.info(f"Estimated coefficients for linear regression: \n{reg_pol.coef_}")
  logging.info(f"Intercept: \n{reg_pol.intercept_}")

  logging.info(
    f"Reg coefficients: \n{pd.DataFrame(reg_pol.coef_, poly_reg.get_feature_names_out(), columns=['reg_coef'])}")

  y_p_train = reg_pol.predict(X_poly_train)
  y_p_test = reg_pol.predict(X_poly_test)

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
  plt.savefig(os.path.join(folder, f'polynomial_reg_scatter.png'))
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
  plt.savefig(os.path.join(folder, f'polynomial_reg_qq_plot.png'))
  plt.show()

  # Uncertanity of prediction
  X2 = sm.add_constant(X_train)
  model = sm.OLS(y_train, X2).fit()
  logging.info(f"Summary of Regression in Statsmodels: \n{model.summary()}")

  # extract p-values for all predictor variables
  # for x in range(0, 3):
  #   logging.info(model.pvalues[x])

  # Cross validation
  kf = KFold(n_splits=10)
  r2_train = []
  r2_valid = []

  for train, valid in kf.split(X_poly_train):
    clf = Lasso(alpha=1)
    clf.fit(X_poly_train[train], y_poly_train[train])

    lasso_train_pred = clf.predict(X_poly_train[train])
    lasso_valid_pred = clf.predict(X_poly_train[valid])

    r2_train.append(r2_score(y_poly_train[train], lasso_train_pred))
    r2_valid.append(r2_score(y_poly_train[valid], lasso_valid_pred))

  plt.scatter(range(0, len(r2_train)), r2_train)
  plt.scatter(range(0, len(r2_train)), r2_valid)
  plt.legend(['train', 'valid'])
  plt.savefig(os.path.join(folder, f'polynomial_reg_cross_validation_scatter.png'))
  plt.show()

  logging.info("Polynomial regression finished.")
