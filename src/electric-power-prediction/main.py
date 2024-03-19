import argparse
import os
import sys
sys.path.append('../')
import pandas as pd
import logging

import eda as eda
from Regression import regression


def main():
  # Create ArgumentParser object
  parser = argparse.ArgumentParser(
    description="This is an electric power prediction application "
                "with different regression technics. ")

  # Add arguments
  parser.add_argument('folder', type=str, help='Output result folder path.')
  parser.add_argument('train_path', type=str, help='Train input file path.')
  parser.add_argument('runt_type', type=str, help='To run an Exploratory data '
                                                  'analysis (EDA) or Regression'
                                                  '(REG) calculation.')
  parser.add_argument('--test_path', type=str, help='Test input file path.')

  # Parse arguments
  args = parser.parse_args()

  return args.folder, args.train_path, args.runt_type, args.test_path


def init_logger(folder, runt_type):
  if not os.path.exists(folder):
    os.makedirs(folder)

  log_file = folder + '/output_' + runt_type + '.log'

  if os.path.exists(log_file):
    os.remove(log_file)

  print(f"Output log can be seen at: {log_file}")
  logging.basicConfig(level=logging.INFO, filename=log_file,
                      format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == '__main__':
  folder, train_path, run_type, test_path = main()
  init_logger(folder, run_type)

  df_train = pd.read_excel(train_path)

  if run_type == 'EDA':
    print("EDA started.")
    eda.EDA_with_pycaret(df_train)
    #eda.analyse_data(folder, df_train)
    print("EDA finished.")
  elif run_type == 'REG':
    print("Regression analysis started.")
    df_test = pd.read_excel(test_path)
    regression.create_models(df_train, df_test,folder)
    # Perform regression analysis
    print("Regression analysis finished.")
  else:
    print("Invalid analysis type.")
    logging.error("Invalid analysis type.")
    exit(1)

  exit()
