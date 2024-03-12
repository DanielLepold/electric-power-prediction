import argparse
import os

import eda as eda
import logging


def main():
  # Create ArgumentParser object
  parser = argparse.ArgumentParser(
    description="This is an electric power prediction application "
                "with different regression technics. ")

  # Add arguments
  parser.add_argument('folder', type=str, help='Output result folder path.')
  parser.add_argument('input_path', type=str, help='Input file path.')

  args = parser.parse_args()
  # Parse arguments
  return args.folder, args.input_path

def init_logger(folder):
  if not os.path.exists(folder):
    os.makedirs(folder)

  log_file=folder+'/output.log'
  print(f"Output log can be seen at: {log_file}")
  logging.basicConfig(level=logging.INFO, filename=log_file,
                      format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    folder,input_path = main()
    print("Electric power prediction started.")
    init_logger(folder)
    print("EDA started.")
    eda.analyse_data(folder,input_path)
    print("Electric power prediction finished.")

    exit()



