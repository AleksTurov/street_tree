import os
import sys
import joblib
import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI
from typing import List
from src.config import PATH_MODELS
from src.modeling import TabularNN
from src.preprocessing import load_and_encode_categorical, split_problems
from src.downloader import downloader_model
from src.utils import logger
from src.model import TreeData, PredictionResponse
from src.preprocessing import load_and_encode_categorical, split_problems
from sklearn.preprocessing import StandardScaler
import uvicorn

# Создаем FastAPI приложение
app = FastAPI(
    title="Predict Health API",
    description="API для предсказания здоровья деревьев",
    version="0.1"
)

# Загружаем модель, кодировщики меток и скейлер
loaded_model, inverse_target_mapping, label_encoders, scaler = downloader_model(PATH_MODELS)
logger.info("Модель успешно загружена!")

@app.post("/predict_health/", response_model=List[PredictionResponse])
async def predict_health(tree_data_list: List[TreeData]):
    """
    Эндпоинт для получения предсказаний здоровья деревьев на основе JSON данных.
    """
    try:
        data = []
        tree_ids = []
        for tree_data in tree_data_list:
            tree_ids.append(tree_data.tree_id)
            data.append(tree_data.dict())

        df = pd.DataFrame(data, index=tree_ids)
        df.index.name = 'tree_id'

        # Предобработка
        df = split_problems(df, created_columns=False)
        df = load_and_encode_categorical(df, list(set(label_encoders.keys())), PATH_MODELS)

        # Проверяем наличие всех необходимых столбцов
        for col in label_encoders.keys():
            if col not in df.columns:
                raise ValueError(f"Отсутствует столбец после кодирования: {col}")

        # Удаляем tree_id, если он есть в столбцах
        if 'tree_id' in df.columns:
            df.drop('tree_id', axis=1, inplace=True)

        # Масштабируем данные
        scaled_data = scaler.transform(df)

        # Делаем предсказания
        predictions = loaded_model.predict(scaled_data)
        probabilities = loaded_model.predict_proba(scaled_data).tolist()
        class_labels = [inverse_target_mapping[label] for label in predictions]

        # Формируем ответ
        results = [
            PredictionResponse(
                tree_id=tree_id,
                predictions=str(pred),  # Convert predictions to string
                probably=prob,
                class_labels=label,
                name_model="TreeHealthModel"
            )
            for tree_id, pred, prob, label in zip(tree_ids, predictions, probabilities, class_labels)
        ]

        return results


    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        return [{"error": str(e)}]

if __name__ == "__main__":
    logger.info("FastAPI is starting...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)
