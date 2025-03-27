import os, sys
import joblib
import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from config import PATH_MODELS, PATH_PROCESSED
from modeling import TabularNN
from downloader import downloader_model

app = FastAPI()

loaded_model, inverse_target_mapping, label_encoders, scaler = dowloader_model(PATH_MODELS)
logger.info("Model loaded successfully!")

# Пример использования:
df = pd.read_csv(f"{PATH_PROCESSED}/test.csv")
df = df.drop('health', axis=1)
df.columns = [col.lower().replace(' ', '_') for col in df.columns]
data_json = df.to_dict(orient='records')
json_data = json.dumps(data_json)

tree_data_list = TreeData.from_dataframe(json_data)

# %%
# Extract tree_id and data from TreeData objects
data = []
tree_ids = []
for tree_data in tree_data_list:
    tree_ids.append(tree_data.tree_id)
    data.append(tree_data.dict())

# Create DataFrame from data
df_reconstructed = pd.DataFrame(data, index=tree_ids)

# Set tree_id as index
df_reconstructed.index.name = 'tree_id'

df_reconstructed.head()

# %%
from src.preprocessing import load_and_encode_categorical, split_problems

data = split_problems(df_reconstructed, created_columns=False)


data = load_and_encode_categorical(data, list(set(label_encoders.keys())), PATH_MODELS)
data.drop('tree_id', axis=1, inplace=True)
# Выгрузим нормальзованные данные
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler = joblib.load(f'{PATH_MODELS}/scaler.pkl')
print("Scaler loaded from scaler.pkl")

data = scaler.transform(data)



# %%
predictions = loaded_model.predict(data)
class_labels = [inverse_target_mapping[label] for label in predictions]
class_labels[:5]

# %%
# Убедимся, что длина предсказаний совпадает с количеством строк в DataFrame
if len(predictions) == len(data) and len(class_labels) == len(data):
    # Добавим предсказания и метки классов в DataFrame
    df_reconstructed['health'] = class_labels
    df_reconstructed.reset_index(inplace=True)

    print(df_reconstructed[['health', 'tree_id']] )
else:
    print("Ошибка: длина предсказаний или меток классов не совпадает с длиной DataFrame.")


