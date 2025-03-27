import os
import sys
import joblib
import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from src.config import PATH_MODELS
from src.modeling import TabularNN
from src.downloader import downloader_model
from src.utils import logger
from src.model import TreeData  # Import TreeData
import json
from src.preprocessing import load_and_encode_categorical, split_problems
from sklearn.preprocessing import StandardScaler
from typing import List

# Создаем FastAPI приложение
app = FastAPI(
    title="Predict Health API",
    description="API для предсказания здоровья деревьев",
    version="0.1"
)

# Загружаем модель, обратное отображение целевых значений, кодировщики меток и скейлер
loaded_model, inverse_target_mapping, label_encoders, scaler = downloader_model(PATH_MODELS)
logger.info("Модель успешно загружена!")

@app.post("/predict_health/", response_model=List[dict])
async def predict_health(tree_data_list: List[TreeData]):
    """
    Эндпоинт для получения предсказаний здоровья деревьев на основе JSON данных.
    """
    try:
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

        # Предварительная обработка
        df = split_problems(df_reconstructed, created_columns=False)
        df = load_and_encode_categorical(df, list(set(label_encoders.keys())), PATH_MODELS)

        # Убедимся, что все ожидаемые столбцы присутствуют
        expected_columns = list(label_encoders.keys())
        for col in expected_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует столбец после кодирования: {col}")

        # Удаляем tree_id, если он существует
        if 'tree_id' in df.columns:
            df.drop('tree_id', axis=1, inplace=True)

        # Получаем названия столбцов перед масштабированием
        before_scaling_columns = df.columns.tolist()

        # Масштабируем данные
        scaled_data = scaler.transform(df)

        # Преобразуем масштабированные данные обратно в DataFrame
        scaled_df = pd.DataFrame(scaled_data, columns=before_scaling_columns)

        # Делаем предсказания
        predictions = loaded_model.predict(scaled_data)
        probably = loaded_model.predict_proba(scaled_data)
        class_labels = [inverse_target_mapping[label] for label in predictions]

        # Добавляем предсказания в DataFrame
        scaled_df['predicted_health'] = class_labels

        # Преобразуем DataFrame в JSON ответ
        result = scaled_df[['']].to_dict(orient="records")
        return result

    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    print("FastAPI is starting...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)