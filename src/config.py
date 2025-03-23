import os
from src.utils import logger

# Путь к основной директории проекта
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Путь к директории с логами
PATH_LOGS = os.path.join(PROJECT_DIR, "logs")
# Путь к директории с данными
PATH_DATA = os.path.join(PROJECT_DIR, "data")
# Путь к директории с исходными данными
PATH_RAW = os.path.join(PATH_DATA, "raw")
PATH_PROCESSED = os.path.join(PATH_DATA, "processed")

# Пути к директориям с моделями, отчетами и документацией
PATH_MODELS = os.path.join(PROJECT_DIR, "models")
PATH_REPORTS = os.path.join(PROJECT_DIR, "reports")
PATH_DOCS = os.path.join(PROJECT_DIR, "docs")

# Путь к директории с исходным кодом
PATH_SRC = os.path.join(PROJECT_DIR, "src")

# URL для загрузки датасета
URL = "https://www.kaggle.com/api/v1/datasets/download/new-york-city/ny-2015-street-tree-census-tree-data"

# Имя датасета
NAME_DATASET = "2015-street-tree-census-tree-data"

# Путь для сохранения скачанного архива
output_path = os.path.join(PATH_RAW, f"{NAME_DATASET}.zip")

logger.info(f"PROJECT_DIR: {PROJECT_DIR}")
logger.info(f"PATH_LOGS: {PATH_LOGS}")
logger.info(f"PATH_DATA: {PATH_DATA}")
logger.info(f"PATH_RAW: {PATH_RAW}")
logger.info(f"PATH_PROCESSED: {PATH_PROCESSED}")
logger.info(f"PATH_MODELS: {PATH_MODELS}")
logger.info(f"PATH_REPORTS: {PATH_REPORTS}")
logger.info(f"PATH_DOCS: {PATH_DOCS}")
logger.info(f"PATH_SRC: {PATH_SRC}")
logger.info(f"output_path: {output_path}")