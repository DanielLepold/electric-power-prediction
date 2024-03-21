import logging
import numpy as np
import statsmodels.api as sm
import sys
import os
from pycaret.regression import RegressionExperiment
import matplotlib.pyplot as plt


def plot_result(error_std, df_test, predictions, folder_path):
  sm.qqplot(error_std, line='45')

  plt.savefig(f"{folder_path}/qq_plot.png")
  plt.show()

  plt.scatter(df_test["PE"], predictions["prediction_label"])
  plt.axline((420, 420), (500, 500), color='black', linewidth=1)
  # Add labels and title
  plt.xlabel("Predicted PE")
  plt.ylabel('Test PE')
  plt.title(f'Scatter Plot of regression result')
  plt.savefig(f"{folder_path}/scatter_plot.png")
  plt.show()


def perform_regression(df_train, df_test, folder):
  print("PyCaret analyses started.")
  folder_path = folder + "/PyCaret"
  # Create the folder if it doesn't exist
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

  # Logging console log to the file.
  original_stdout = sys.stdout
  log_file_path = folder_path + '/output_pycaret.log'

  if os.path.exists(log_file_path):
    os.remove(log_file_path)

  sys.stdout = open(log_file_path, 'a')

  print("-----------------------------------------")
  print("PyCaret analyses started.")

  # Init setup
  s = RegressionExperiment()
  s.setup(data=df_train, test_data=df_test, target="PE", index=False,
          session_id=123)

  # Model training and selection
  best = s.compare_models()
  # logging.info(f"Result of compare models: \n{best} ")

  # Evaluate trained model
  s.evaluate_model(best)

  # Predict on hold-out/test set
  pred_holdout = s.predict_model(best)
  # logging.info(f"Prediction based on the best model: \n{pred_holdout} ")

  # Predict on new data
  new_data = df_test.copy().drop("PE", axis=1)
  predictions = s.predict_model(best, data=new_data)

  error = predictions["prediction_label"] - df_test["PE"]
  error_std = (error - np.mean(error)) / np.std(error)

  plot_result(error_std, df_test, predictions, folder_path)

  # Adjusting back the console output.
  sys.stdout = original_stdout
  print("PyCaret analyses finished.")
