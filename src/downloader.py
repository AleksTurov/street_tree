
import pandas as pd
from src.config import PATH_MODELS, PATH_PROCESSED
import os, sys
import joblib
import torch
from src.modeling import TabularNN
from src.utils import logger
import numpy as np

def downloader_model(PATH_MODELS):
    # Пути к моделям и энкодерам
    model_path = f"{PATH_MODELS}/tabular_model.pth"
    scaler_path = f"{PATH_MODELS}/scaler.pkl"
    encoders_path = f"{PATH_MODELS}/label_encoders.pkl"

    target_encoder_path = f"{PATH_MODELS}/label_encoders_target.pkl"
    target_mapping = joblib.load(target_encoder_path)
    inverse_target_mapping = {v: k for k, v in target_mapping.items()}
    logger.info(f"{inverse_target_mapping} - target mapping")

    # Load the checkpoint to get the parameters
    checkpoint = torch.load(model_path)
    input_dim = checkpoint['input_dim']
    hidden_dims = checkpoint['hidden_dims']
    output_dim = checkpoint['output_dim']
    target_mapping = checkpoint['target_mapping']

    # Create dummy data with the correct shape
    dummy_X = pd.DataFrame(np.zeros((3, input_dim)))
    dummy_y = pd.Series([0, 1, 2]) 

    # Initialize the model using dummy data
    loaded_model = TabularNN(dummy_X, dummy_y, dummy_X, dummy_y, hidden_dims=hidden_dims)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    logger.info("Model loaded successfully!")

    # Обратное отображение для целевого признака
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoders_path)
    target_mapping = joblib.load(target_encoder_path)
    inverse_target_mapping = {v: k for k, v in target_mapping.items()}
    return loaded_model, inverse_target_mapping, label_encoders, scaler