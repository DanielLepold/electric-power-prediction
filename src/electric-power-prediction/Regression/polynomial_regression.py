from sklearn.preprocessing import PolynomialFeatures
import os
import sys
import logging
import pandas as pd
sys.path.insert(0, '/Regression/')

def create_model(df_train,df_test,output_folder_path):
  print("Polynomial regression.")
