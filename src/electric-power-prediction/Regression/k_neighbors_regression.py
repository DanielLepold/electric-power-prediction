import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
from .base import calculate_errors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict


def perform_regression(X_train, X_test, y_train, y_test, folder,df_train):
  logging.info("\n-------------------------------------------------------------"
               "--------------------------------")
  logging.info("K neighbors regression started.")
  print("K neighbors regression started.")

  neighbors = range(2, 21)  # Különböző szomszédok száma 2-től 20-ig

  results_n = {'train': [], 'test': []}

  for n in neighbors:
    k_neighbor_reg = KNeighborsRegressor(n_neighbors=n)
    k_neighbor_reg.fit(X_train, y_train)

    # Cross-validation
    y_p_train = cross_val_predict(k_neighbor_reg, X_train, y_train, cv=5)
    y_p_test = k_neighbor_reg.predict(X_test)

    # Calculating R^2 for cross-validation
    mae_train, mse_train, r2_train_cv = calculate_errors(y_train, y_p_train)
    mae_test, mse_test, r2_test_cv = calculate_errors(y_test, y_p_test)

    results_n['train'].append(r2_train_cv)
    results_n['test'].append(r2_test_cv)

  df_results = pd.DataFrame(results_n).T
  logging.info(f"Errors: \n{df_results}")


  plt.plot(neighbors, results_n['train'], label='Train R2_CV')
  plt.plot(neighbors, results_n['test'], label='Test R2')
  plt.xlabel('Number of Neighbors')
  plt.ylabel('R^2 Score')
  plt.title('Cross-validation R^2 Score vs Number of Neighbors')
  plt.legend()
  plt.savefig(os.path.join(folder, 'knn_regression_cv.png'))
  plt.show()


  # N = 5 -nal volt a legjobb eredmeny
  k_neighbor_reg = KNeighborsRegressor(n_neighbors=5)
  k_neighbor_reg.fit(X_train, y_train)

  # Cross-validation
  y_p_train = cross_val_predict(k_neighbor_reg, X_train, y_train, cv=5)
  y_p_test = k_neighbor_reg.predict(X_test)

  # Calculating R^2 for cross-validation
  mae_train, mse_train, r2_train = calculate_errors(y_train, y_p_train)
  mae_test, mse_test, r2_test = calculate_errors(y_test, y_p_test)

  results = {}
  results["train"] = {'MAE': mae_train, 'MSE': mse_train,
                      'R2': r2_train}
  results["test"] = {'MAE': mae_test, 'MSE': mse_test, 'R2': r2_test}

  df_results = pd.DataFrame(results).T
  logging.info(f"Errors: \n{df_results}")

  # Scatter plot for test vs prediction
  plt.scatter(y_test, y_p_test)
  plt.axline((420, 420), (500, 500), color='black', linewidth=1)
  plt.xlabel("y_test")
  plt.ylabel('y_p_test')
  plt.savefig(os.path.join(folder, f'k_neighbor_reg_scatter.png'))
  plt.show()

  logging.info("K neighbors regression finished.")

  return mae_test, mse_test, r2_test, 'K Neighbors Regressor'


