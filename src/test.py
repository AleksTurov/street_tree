import json
import pandas as pd
from src.config import PATH_PROCESSED
from src.model import TreeData


df = pd.read_csv(f"{PATH_PROCESSED}/test.csv")
df = df.drop('health', axis=1)
df.columns = [col.lower().replace(' ', '_') for col in df.columns]
data_json = df.to_dict(orient='records')
json_data = json.dumps(data_json)

tree_data_list = TreeData.from_dataframe(json_data)
