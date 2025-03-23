import pandas as pd
import numpy as np
from src.utils import logger
import re
import os
import pickle   
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def df_fillna(df):
    """
    Заполнение пропусков в датафрейме.
    """
    df['spc_latin'].fillna('No observation',inplace=True)
    df['sidewalk'].fillna('NoDamage',inplace=True)
    df['problems'].fillna('NoProblem',inplace=True)
    df['steward'].fillna('None',inplace=True)
    df['guards'].fillna('Unsure',inplace=True)
    logger.info("Пропуски заполнены")
    return df

def split_problems(df, created_columns=True):
    """
    Функция для разделения строки с проблемами на отдельные проблемы,
    создания новых колонок для каждой проблемы и подсчета количества проблем.
    """
    
    # Разделение строки с проблемами на отдельные проблемы
    df['problems_new'] = df['problems'].str.split(',')
    
    # Получение уникальных проблем
    unique_problems = set(problem.strip().lower().replace(' ', '_') for sublist in df['problems_new'] for problem in sublist)
    logger.info(f'{unique_problems} - уникальные проблемы')
    
    # Создание новых колонок
    if created_columns:
        for problem in unique_problems:
            df[problem] = df['problems'].str.contains(problem, case=False)
    
    # Подсчет количества проблем
    df['num_problems'] = df['problems_new'].apply(len)
    
    df.drop(columns=['problems', 'problems_new'], inplace=True)

    return df


def convert_to_bool(df):
    """
    Преобразование значений в столбцах в булевые значения
    curb_loc - OnCurb -> True, OffCurb -> False
    sidewalk - Damage -> True, NoDamage -> False
    root_stone, root_grate, root_other, trunk_wire, trnk_light, trnk_other, brch_light, brch_shoe, brch_other
    Yes -> True, No -> False
    """   
    df['curb_loc'] = df['curb_loc'].apply(lambda x: 1 if x == 'OnCurb' else 0)

    df['sidewalk'] = df['sidewalk'].apply(lambda x: 1 if x == 'Damage' else 0)
    
    # Преобразование других столбцов в булевые значения
    columns_to_convert = [
        'root_stone', 'root_grate', 'root_other', 'trunk_wire', 
        'trnk_light', 'trnk_other', 'brch_light', 'brch_shoe', 'brch_other'
    ]

    for col in columns_to_convert:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)


    logger.info("Значения преобразованы в булевые")
    return df


def encode_and_save_categorical(df, categorical_columns, model_path, name_labe_encoders):
    """
    Преобразует категориальные признаки в числовые значения, сохраняет LabelEncoders.

    Args:
        df (pd.DataFrame): DataFrame для преобразования.
        categorical_columns (list): Список категориальных колонок.
        model_path (str): Путь для сохранения LabelEncoders.  Defaults to '../models'.

    Returns:
        pd.DataFrame: Преобразованный DataFrame.
        dict: Словарь LabelEncoder'ов.
    """
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    # Ensure the directory exists
    os.makedirs(model_path, exist_ok=True)

    # Save the encoders
    with open(os.path.join(model_path, name_labe_encoders), 'wb') as f:
        pickle.dump(label_encoders, f)

    return df, label_encoders


def load_and_encode_categorical(df, categorical_columns, model_path):
    """
    Загружает сохраненные LabelEncoders и преобразует категориальные признаки.

    Args:
        df (pd.DataFrame): DataFrame для преобразования.
        categorical_columns (list): Список категориальных колонок.
        model_path (str): Путь, где сохранены LabelEncoders.

    Returns:
        pd.DataFrame: Преобразованный DataFrame.
    """
    encoders_path = os.path.join(model_path, 'label_encoders.pkl')

    if not os.path.exists(encoders_path):
        logger.error(f"Файл с LabelEncoders не найден по пути: {encoders_path}")
        return df  # Возвращаем DataFrame без изменений

    with open(encoders_path, 'rb') as f:
        label_encoders = pickle.load(f)

    for column in categorical_columns:
        le = label_encoders.get(column)
        if le is None:
            logger.warning(f"LabelEncoder не найден для столбца: {column}")
            continue  # Пропускаем этот столбец
        # Преобразуем значения в колонке с помощью LabelEncoder
        df[column] = le.transform(df[column])

    logger.info("Категориальные признаки преобразованы с использованием загруженных LabelEncoders")
    return df


def split_and_save(X, y, output_dir, size, name_train, name_test):
    """ 
    Рзделим данные на обучающую, валидационную и тестовую выборки и сохраним их в output_dir
    stratify=y - сохранение пропорции классов в выборках
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42, stratify=y)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    os.makedirs(output_dir, exist_ok=True)
    X_train.assign(health=y_train).to_csv(f"{output_dir}/{name_train}", index=False)
    X_test.assign(health=y_test).to_csv(f"{output_dir}/{name_test}", index=False)

    logger.info("Data successfully saved to: %s", output_dir)
    logger.info("Train data shape: %s", X_train.shape)
    logger.info("Test data shape: %s", X_test.shape)

    return X_train, X_test, y_train, y_test

