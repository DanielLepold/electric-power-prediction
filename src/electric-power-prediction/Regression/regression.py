import logging
import os
import sys
sys.path.insert(0, 'Regression')

import pandas as pd
from Regression import simple_linear_regression as slr
from Regression import polynomial_regression as pr
from Regression import regression_pycaret as rp


def create_models(df_train, df_test,folder):
  logging.info("REG - Regression model generation started.")

  folder_path = folder + "/Regression"
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

  slr.create_model(df_train, df_test, folder_path)
  pr.create_model(df_train, df_test, folder_path)
  rp.create_model(df_train,df_test,folder_path)

  logging.info("REG - Regression model generation finished.")
