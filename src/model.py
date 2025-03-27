import pandas as pd
from pydantic import BaseModel, Field

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from typing import List, Union
import numpy as np
import json


class TreeData(BaseModel):
    """
    Pydantic модель для данных о деревьях.
    """
    tree_id: int = Field(..., description="Идентификатор дерева.")
    block_id: int = Field(..., description="Идентификатор блока, к которому относится дерево.")
    tree_dbh: int = Field(..., description="Диаметр дерева на высоте груди.")
    curb_loc: int = Field(..., description="Местоположение дерева относительно бордюра.")
    spc_latin: str = Field(..., description="Латинское название дерева.")
    steward: str = Field(..., description="Информация о попечителе.")
    guards: str = Field(..., description="Информация о ограждении.")
    sidewalk: int = Field(..., description="Повреждения тротуара рядом с деревом.")
    user_type: str = Field(..., description="Тип пользователя.")
    problems: str = Field(..., description="Проблемы с деревом.")
    root_stone: int = Field(..., description="Информация о камне в корнях.")
    root_grate: int = Field(..., description="Информация о решетке для корней.")
    root_other: int = Field(..., description="Дополнительная информация о корнях.")
    trunk_wire: int = Field(..., description="Информация о проволоке на стволе.")
    trnk_light: int = Field(..., description="Информация о освещении на стволе.")
    trnk_other: int = Field(..., description="Дополнительная информация о стволе.")
    brch_light: int = Field(..., description="Информация о освещении на ветках.")
    brch_shoe: int = Field(..., description="Информация о наличии обуви на ветках.")
    brch_other: int = Field(..., description="Дополнительная информация о ветках.")
    postcode: int = Field(..., description="Почтовый индекс.")
    borough: str = Field(..., description="Район города.")
    cncldist: int = Field(..., description="Номер района города.")
    st_assem: int = Field(..., description="Идентификатор представителя ассамблеи.")
    st_senate: int = Field(..., description="Идентификатор сенатора штата.")
    nta: str = Field(..., description="Идентификатор соседства.")
    boro_ct: int = Field(..., description="Идентификатор района.")
    latitude: float = Field(..., description="Широта местоположения дерева.")
    longitude: float = Field(..., description="Долгота местоположения дерева.")

    class Config:
        extra = "forbid"  # Запрещает добавление дополнительных данных, не указанных в модели.

    @staticmethod
    def convert_to_bool(df: pd.DataFrame):
        """
        Convert certain columns to boolean values (represented as 1 or 0).
        Convert 'Yes'/'No' values to 1/0 for the root and trunk-related columns
        """
        # Convert 'curb_loc' to 1 if 'OnCurb', 0 otherwise
        df['curb_loc'] = df['curb_loc'].apply(lambda x: 1 if x == 'OnCurb' else 0)
        df['sidewalk'] = df['sidewalk'].apply(lambda x: 1 if x == 'Damage' else 0)

        # Convert 'Yes'/'No' values to 1/0 for the root and trunk-related columns
        columns_to_convert = [
            'root_stone', 'root_grate', 'root_other', 'trunk_wire', 
            'trnk_light', 'trnk_other', 'brch_light', 'brch_shoe', 'brch_other'
        ]
        for col in columns_to_convert:
            df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # Ensure boolean columns are of integer type
        bool_cols = ['curb_loc', 'sidewalk', 'root_stone', 'root_grate', 'root_other', 'trunk_wire', 'trnk_light', 'trnk_other', 'brch_light', 'brch_shoe', 'brch_other']
        for col in bool_cols:
            df[col] = df[col].astype(int)

        return df

    @staticmethod
    def df_fillna(df: pd.DataFrame):
        """
        Fill missing values in the DataFrame.
        """
        df['spc_latin'].fillna('No observation', inplace=True)
        df['sidewalk'].fillna(0, inplace=True)  # Assuming numeric 0 for no damage
        df['problems'].fillna('NoProblem', inplace=True)
        df['steward'].fillna('None', inplace=True)
        df['guards'].fillna('Unsure', inplace=True)
        return df

    @classmethod
    def from_dataframe(cls, data: Union[pd.DataFrame, str]) -> List["TreeData"]:
        """
        Convert a DataFrame or JSON string to a list of TreeData objects.
        """
        if isinstance(data, str):  # If input is a JSON string
            data = json.loads(data)  # Load JSON data
            df = pd.DataFrame(data)  # Convert to DataFrame
        else:
            df = data.copy()  # Create a copy to avoid modifying the original DataFrame

        df = cls.convert_to_bool(df)  # Convert columns to boolean values
        df = cls.df_fillna(df)  # Fill missing values
        # Only keep the columns defined in the model
        columns = set(cls.model_fields.keys())  # Update for Pydantic v2
        df = df[list(columns)]
        data_json = df.to_dict(orient='records')  # Convert to list of dictionaries
        return [cls(**data) for data in data_json]  # Convert to TreeData objects
class PredictionResponse(BaseModel):
    tree_id: int = Field(..., description="Идентификатор дерева.")
    predictions: str = Field(..., description="Предсказанный класс.")
    probably: List[float] = Field(..., description="Вероятности классов.")
    class_labels: str = Field(..., description="Метки класс для предсказания.")
    name_model: str = Field(..., description="Название модели.")
