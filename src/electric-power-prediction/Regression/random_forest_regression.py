import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from .base import calculate_errors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV


def plot_grid_search(folder, cv_results, grid_param_1, grid_param_2, name_param_1,
                     name_param_2):
  # Get Test Scores Mean and std for each grid search
  scores_mean = cv_results['mean_test_score']
  scores_mean = np.array(scores_mean).reshape(len(grid_param_2),
                                              len(grid_param_1))

  scores_sd = cv_results['std_test_score']
  scores_sd = np.array(scores_sd).reshape(len(grid_param_2), len(grid_param_1))

  # Plot Grid search scores
  _, ax = plt.subplots(1, 1)

  # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
  for idx, val in enumerate(grid_param_2):
    ax.plot(grid_param_1, scores_mean[idx, :], '-o',
            label=name_param_2 + ': ' + str(val))

  ax.set_title("Grid Search Scores")
  ax.set_xlabel(name_param_1)
  ax.set_ylabel('CV R2 Score')
  ax.legend(loc="best")
  ax.grid('on')

  # Save the plot
  plt.savefig(os.path.join(folder, f'grid_search_result.png'))
  plt.show()


def grid_analysis(X_train, X_test, y_train, y_test, folder):
  logging.info("\n-------------------------------------------------------------"
               "--------------------------------")
  logging.info("Random forest regression started.")

  # RandomForestRegressor inicializálása
  random_reg = RandomForestRegressor(random_state=42)

  # Grid Search paraméterek definiálása
  param_grid = {
    'n_estimators': [10,20,50, 100, 150, 200,250,300,500],
    'max_depth': [None]
  }

  grid_search = GridSearchCV(estimator=random_reg, param_grid=param_grid, cv=5,
                             scoring='r2')

  grid_search.fit(X_train, y_train)

  print("Legjobb paraméterek:", grid_search.best_params_)
  print("Legjobb R^2:", grid_search.best_score_)

  plot_grid_search(folder,grid_search.cv_results_, param_grid['n_estimators'], param_grid['max_depth'],
                   'n_estimators', 'max_depth')

  #result: n_estimators=100 gives us good enough result, max_depth=None
  #First case:
  # param_grid = {
  #   'n_estimators': [10,20,50, 100, 150, 200,250],
  #   'max_depth': [None,5,10,15,20,25]
  # }
  #Second case:
  # param_grid = {
  #   'n_estimators': [10,20,50, 100, 150, 200,250,300,500],
  #   'max_depth': [None]
  # }


  logging.info("Random forest regression finished.")

def perform_regression(X_train, X_test, y_train, y_test, folder,df_train):
  logging.info("\n-------------------------------------------------------------"
               "--------------------------------")
  logging.info("Random forest regression started.")
  print("Random forest regression started.")

  random_reg = RandomForestRegressor(n_estimators=100, max_depth=None,
                                     random_state=42)
  random_reg.fit(X_train, y_train)

  # Cross-validation
  y_p_train = cross_val_predict(random_reg, X_train, y_train,
                                   cv=5)  # K=5 cross-validation
  y_p_test = random_reg.predict(X_test)

  # Calculating errors for train and test sets
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
  plt.savefig(os.path.join(folder, f'random_tree_reg_scatter.png'))
  plt.show()

  logging.info("Random forest regression finished.")

  return mae_test, mse_test, r2_test, 'Random Forest Regressor'


