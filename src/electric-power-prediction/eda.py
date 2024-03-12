import pandas as pd
import table as tb
import plot as pl
import logging

def analyse_data(folder,input_path):
  logging.info("EDA - Exploratory data analysis started.")
  df = pd.read_excel(input_path)

  logging.info(f"Creating plots before data clearance at: {folder}/Initial/Plots")
  pl.create_plots(df, folder+"/Initial")

  tb.create_table(df.describe(), folder, "/description.xlsx")

  logging.info(f"Non unique rows: \n{df.loc[:, ].nunique()}")

  logging.info(f"Data Framework shape: \n{df.shape}")

  logging.info(f"Duplicated rows shape: \n{df[df.duplicated()].shape}")

  logging.info(f"Duplicated rows: \n{df[df.duplicated()]}")

  # Dropping duplicates
  logging.info("Dropping duplicates")
  df = df.drop_duplicates()

  logging.info(f"Df shape: \n{df.shape}")

  logging.info(f"Sum of null rows: \n{df.isnull().sum()}")

  Q1 = df.quantile(0.25)
  logging.info(f"Q1, quantile(0.25), the lower hinge: \n{Q1}")

  Q3 = df.quantile(0.75)
  logging.info(f"Q3, quantile(0.75), the upper hinge: \n{Q3}")

  IQR = Q3 - Q1
  logging.info(f"IQR, quantile(0.5), the mean: \n{IQR}")

  UW = Q3 + 1.5 * IQR
  logging.info(f"UW, Upper whisker end: \n{UW}")

  LW = Q1 - 1.5 * IQR
  logging.info(f"LW, Lower whisker end: \n{LW}")

  logging.info("Removing outliers from dataset.")
  df = df[~((df < LW) | (df > UW)).any(axis=1)]

  logging.info(f"Creating plots after data clearance at: {folder}/Plots")
  pl.create_plots(df, folder)

  logging.info(f"Writing cleared data result into excel.")
  tb.create_table(df,folder,"/cleared.xlsx")
  logging.info("EDA - Exploratory data analysis finished.")
