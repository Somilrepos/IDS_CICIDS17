import os
import sys

import numpy as np 
import pandas as pd
import pickle5 as pickle
import yaml

from src.exception import CustomException


def getData(path):
  files = os.listdir(path)
  files = [f for f in files if os.path.isfile(path+'/'+f)]

  dfs = {}
  for file in files:
      dfs[file] = pd.read_csv(path+"/"+file)

  #Combine all the datasets
  df = pd.DataFrame()

  for filename,data in dfs.items():
      data["WeekDay"] = np.char.array([filename.split("-")[0]]*len(data))
      data["TimeOfDay"] = np.char.array([re.findall(r"[\w']+", filename)[2]]*len(data))
      df = pd.concat([df,data],ignore_index=True)

  return df


def cleanData(df):
    tempdf = df[(df["Flow Bytes/s"] == np.inf) | (df["Flow Bytes/s"].isna())]
    df.drop(tempdf.index, inplace=True)
    return df

def choose(label):
  if label=='BENIGN':
    return 0
  else:
    return 1
  
def save_object(file_path, obj):
  try:
      dir_path = os.path.dirname(file_path)

      os.makedirs(dir_path, exist_ok=True)

      with open(file_path, "wb") as file_obj:
          pickle.dump(obj, file_obj)

  except Exception as e:
      raise CustomException(e, sys)
  

YAML_CONFIG_PATH = os.path.join("config","config.yaml")

def load_config():
    with open(YAML_CONFIG_PATH) as file:
        config = yaml.safe_load(file)
    return config