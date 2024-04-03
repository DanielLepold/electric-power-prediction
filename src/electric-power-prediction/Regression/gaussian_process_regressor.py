import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from .base import calculate_errors

def perform_regression(X_train,X_test, y_train,y_test, folder):
  logging.info("\n-------------------------------------------------------------"
               "--------------------------------")
  logging.info("Gaussian Process regression started.")

  gauss_proc = GaussianProcessRegressor()
  gauss_proc.fit(X_train, y_train)

  r2 = gauss_proc.score(X_train, y_train)
  logging.info(f"Gaussian process score, r2: \n{r2}")

  logging.info("Gaussian Process regression finished.")
