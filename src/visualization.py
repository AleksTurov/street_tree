import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import logger


def plot_corr_matrix(df, method='pearson', threshold=0.3, figsize=(20, 20)):
    """
    Построение матрицы корреляции с фильтрацией значимых корреляций.
    Исключает столбцы, которые не имеют значимых корреляций.
    """
    # Оставляем только числовые столбцы
    numeric_df = df.select_dtypes(include=[np.number, bool])
    
    if numeric_df.empty:
        logger.info("Нет числовых данных для корреляции.")
        return
    
    # Удаляем столбцы с низкой вариативностью
    low_var_cols = [col for col in numeric_df.columns if numeric_df[col].nunique() <= 1]
    if low_var_cols:
        logger.info(f"Удаляем столбцы с низкой вариативностью: {low_var_cols}")
        numeric_df.drop(columns=low_var_cols, inplace=True)
    
    if numeric_df.empty:
        logger.info("После фильтрации нет данных для корреляции.")
        return
    
    # Вычисляем корреляционную матрицу
    corr = numeric_df.corr(method=method)
        
    # Обнуляем корреляции ниже порога
    corr[corr.abs() < threshold] = np.nan

    # Убираем строки и столбцы без значимой корреляции
    corr.dropna(how='all', axis=0, inplace=True)
    corr.dropna(how='all', axis=1, inplace=True)

    if corr.empty:
        logger.info("Нет значимых корреляций по выбранному методу.")
        return
    
    # Построение тепловой карты
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    plt.title(f'Корреляционная матрица ({method})')
    plt.show()
