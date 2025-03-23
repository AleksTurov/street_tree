from pathlib import Path
import os
import zipfile
import requests
from src.utils import logger

def download_file(url, path_raw, output_path):
    """ 
    Загрузка файла по URL и сохранение в output_path
    """
    # Проверка наличия архива
    if not os.path.exists(output_path):
        # Загрузка файла
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=128):
                    f.write(chunk)
            logger.info(f"Файл загружен: {output_path}")
        else:
            logger.info(f"Ошибка при загрузке файла: {response.status_code}")
    else:
        logger.info(f"Файл уже существует: {output_path}")

    # Распаковка архива
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(path_raw)
        logger.info("Файлы распакованы")