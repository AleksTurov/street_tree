from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import os
from src.utils import logger


class CatBoostModelTrain:
    """
    Класс для обучения и оценки модели CatBoost.
    """
    def __init__(self, model_dir="models"):
        """
        Инициализация модели CatBoost.
        Args:
            model_dir (str, optional): Директория для сохранения моделей. Defaults to "models".
        """
        self.model = CatBoostClassifier(iterations=1000, depth=8, learning_rate=0.03, loss_function='MultiClass', verbose=100)
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, X_train, y_train, X_val, y_val):
        """
        Обучение модели CatBoost.
        Args:
            X_train (pd.DataFrame): Обучающие признаки.
            y_train (pd.Series): Обучающие метки.
            X_val (pd.DataFrame): Валидационные признаки.
            y_val (pd.Series): Валидационные метки.
        """
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=False)
        self._save_model("catboost_model.cbm")

        y_pred = self.model.predict(X_val)
        y_probs = self.model.predict_proba(X_val)

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred, average='weighted'),
            'auc_roc': roc_auc_score(y_val, y_probs, multi_class='ovr'),
            'conf_matrix': confusion_matrix(y_val, y_pred)
        }

        logger.info(f"CatBoost Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"Confusion Matrix:\n{metrics['conf_matrix']}")

    def _save_model(self, filename):
        """
        Сохранение модели CatBoost в файл.
        Args:
            filename (str): Имя файла для сохранения модели.
        """
        path = os.path.join(self.model_dir, filename)
        self.model.save_model(path)
        logger.info(f"Сохраняем модель в {path}")