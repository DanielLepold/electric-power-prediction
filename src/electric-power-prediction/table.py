import os
import pandas as pd
import logging

def create_table(data,output_path,file_name, index=True):
  folder_path=output_path+"/Tables"
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

  df = pd.DataFrame(data)
  df.to_excel(folder_path+file_name, index=index)
  logging.info(f"Table was created at:{folder_path+file_name}.")
