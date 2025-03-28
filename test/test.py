
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(PROJECT_DIR)
import pandas as pd
from src.model import TreeData
from src.config import PATH_PROCESSED
import json
from sklearn.metrics import roc_auc_score
import numpy as np
import requests
import json
from src.model import TreeData
from src.config import PATH_PROCESSED
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer


data1 = pd.read_csv(f"{PATH_PROCESSED}/test.csv")
df = data1.drop('health', axis=1)
df.columns = [col.lower().replace(' ', '_') for col in df.columns]
data_json = df.to_dict(orient='records')
json_data = json.dumps(data_json)

tree_data_list = TreeData.from_dataframe(json_data)

data = [tree_data.dict() for tree_data in tree_data_list]

json_data = json.dumps(data)


url = "http://localhost:8000/predict_health/"
headers = {"Content-Type": "application/json"}
response = requests.post(url, data=json_data, headers=headers)

print(response) # <Response [200]>
# Истинные метки
true_labels = data1['health'].map({'Poor': 0, 'Fair': 1, 'Good': 2}).tolist()

# Извлекаем вероятности из ответа API
predicted_probabilities = np.array([item['probably'] for item in response.json()])

# Рассчитываем AUC-ROC
roc_auc = roc_auc_score(true_labels, predicted_probabilities, multi_class='ovr')

# Выводим результат

print(f"AUC-ROC: {roc_auc}")


