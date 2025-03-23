import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from src.utils import logger
import numpy as np
from focal_loss import FocalLoss

class DeepLearningModel:
    """
    📊 Обоснование выбора архитектуры:

    Мы используем Multilayer Perceptron (MLP) — многослойную нейронную сеть, которая подходит для обработки табличных данных.

    1. Характер данных:
        - Данные о деревьях (например, tree_dbh, borough, curb_loc) — табличные числовые и категориальные признаки.
        - MLP хорошо справляется с такими данными, когда признаки независимы и не имеют пространственной структуры.

    2. Сложность задачи:
        - Это многоклассовая классификация (Good, Fair, Poor).
        - MLP с CrossEntropyLoss и LogSoftmax подходит для решения таких задач.

    🏗️ Архитектура модели:
    - Входной слой (input_size): Размерность равна числу признаков.
    - Скрытые слои:
        - 256, 128, 64 – оптимально для сложных зависимостей и достаточной глубины.
        - BatchNorm – нормализует выходы и ускоряет сходимость.
        - Dropout (0.3) – предотвращает переобучение.
    - Выходной слой:
        - LogSoftmax(dim=1) – преобразует выходы в вероятности для многоклассовой классификации.

    📈 Оптимизация и улучшение обучения:
    - Class Weights – учитывает дисбаланс классов в функции потерь (CrossEntropyLoss).
    - Adam Optimizer – адаптивный метод оптимизации с learning_rate=0.0005.
    - ReduceLROnPlateau – снижает learning_rate, если модель перестает улучшаться (на val_loss).
    """

    def __init__(self, input_size, hidden_layers, output_size=3, learning_rate=0.0005, model_dir="models", y_train=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Используем устройство: {self.device}")

        self.model = self._build_model(input_size, hidden_layers, output_size).to(self.device)

        # Рассчитываем веса классов для балансировки
        if y_train is not None:
            y_train = np.array(y_train, dtype=int)
            if len(y_train) > 0 and len(np.unique(y_train)) > 1:
                class_counts = np.bincount(y_train)
                class_weights = 1.0 / class_counts
                weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
                logger.info(f"Веса классов: {class_weights}")
            else:
                logger.error("Недостаточно классов для расчёта весов.")
                weights = None
        else:
            weights = None
            logger.error("Некорректный формат y_train: проверь тип и форму.")

    
        self.criterion = FocalLoss(alpha=class_weights, gamma=2)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Добавляем scheduler для адаптивного изменения learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        # Директория для сохранения моделей
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def _build_model(self, input_size, hidden_layers, output_size):
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.BatchNorm1d(hidden_layers[0]), nn.ReLU(), nn.Dropout(0.3)]
        for in_size, out_size in zip(hidden_layers[:-1], hidden_layers[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.BatchNorm1d(out_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.4))
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        layers.append(nn.LogSoftmax(dim=1))
        return nn.Sequential(*layers)

    def _prepare_data(self, X, y):
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values, dtype=torch.long).to(self.device)
        return TensorDataset(X_tensor, y_tensor)

    def train(self, X_train, y_train, X_val, y_val, num_epochs=100, batch_size=64):
        logger.info("Начинаем обучение модели глубокого обучения")

        train_dataset = self._prepare_data(X_train, y_train)
        val_dataset = self._prepare_data(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            train_loss = self._train_one_epoch(train_loader)
            val_loss, metrics = self._validate(val_loader)

            logger.info(f"Эпоха {epoch + 1}/{num_epochs}, Потеря при обучении: {train_loss:.4f}, Потеря при валидации: {val_loss:.4f}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
            logger.info(f"Confusion Matrix:\n{metrics['conf_matrix']}")

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model("best_model.pth")

        logger.info("Обучение завершено")

    def _train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate(self, val_loader):
        self.model.eval()
        total_val_loss = 0

        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                total_val_loss += loss.item()

                probs = torch.exp(outputs).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_score': f1_score(all_labels, all_preds, average='weighted'),
            'auc_roc': roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr'),
            'conf_matrix': confusion_matrix(all_labels, all_preds)
        }

        return total_val_loss / len(val_loader), metrics

    def _save_model(self, filename):
        path = os.path.join(self.model_dir, filename)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Сохраняем модель в {path}")