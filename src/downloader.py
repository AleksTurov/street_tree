import os
import joblib
import torch
import pandas as pd
import numpy as np
from src.config import PATH_MODELS
from src.modeling import TabularNN
from src.utils import logger

def downloader_model(PATH_MODELS):
    try:
        # Пути к моделям и энкодерам
        model_path = os.path.join(PATH_MODELS, "tabular_model.pth")
        scaler_path = os.path.join(PATH_MODELS, "scaler.pkl")
        encoders_path = os.path.join(PATH_MODELS, "label_encoders.pkl")
        target_encoder_path = os.path.join(PATH_MODELS, "label_encoders_target.pkl")

        # Загрузка маппинга целевой переменной (если файл существует)
        if os.path.exists(target_encoder_path):
            target_mapping = joblib.load(target_encoder_path)
            inverse_target_mapping = {v: k for k, v in target_mapping.items()}
            logger.info(f"{inverse_target_mapping} - target mapping")
        else:
            logger.warning(f"Target encoder file {target_encoder_path} not found.")
            inverse_target_mapping = None

        # Загрузка модели
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        input_dim = checkpoint.get('input_dim')
        hidden_dims = checkpoint.get('hidden_dims', [])
        output_dim = checkpoint.get('output_dim')

        if not all([input_dim, hidden_dims, output_dim]):
            raise ValueError("Checkpoint is missing required keys!")

        # Создание заглушки данных
        dummy_X = pd.DataFrame(np.zeros((3, input_dim)))
        dummy_y = pd.Series(range(output_dim))  # Учитываем размер output_dim

        # Инициализация модели
        loaded_model = TabularNN(dummy_X, dummy_y, dummy_X, dummy_y, hidden_dims=hidden_dims)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model.eval()

        logger.info("Model loaded successfully!")

        # Загрузка скейлера и энкодеров (если файлы существуют)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        label_encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else None

        return loaded_model, inverse_target_mapping, label_encoders, scaler

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None, None, None
