import pandas as pd
import numpy as np
from src.utils import logger
import re
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib

class TabularNN(nn.Module):
    """
    Описание архитектуры:

    1. Входной слой:
    - Модель принимает табличные данные на вход, где каждая строка представляет образец, а столбцы — признаки.

    2. Скрытые слои:
    - Используются полносвязные слои (Linear) с Batch Normalization и ReLU активацией для выявления сложных взаимодействий между признаками.
    - Batch Normalization стабилизирует обучение и ускоряет сходимость.
    - Dropout применяется для регуляризации и предотвращения переобучения.

    3. Выходной слой:
    - Последний Linear-слой генерирует логиты, которые представляют необработанные выходы модели.
    - Для получения вероятностей классов логиты преобразуются с помощью функции softmax.

    4. Стратегия обучения:
    - Используется CrossEntropyLoss с весами классов для учета дисбаланса классов.
    - Модель отслеживает метрику AUC-ROC и сохраняет лучшую модель по её наибольшему значению.
    - Раннее прекращение (early stopping) используется для предотвращения переобучения и экономии вычислительных ресурсов.

    Эта архитектура подходит для табличных данных с несколькими классами и эффективно обрабатывает дисбаланс классов.
    """
    def __init__(self, X_train, y_train, X_val, y_val, hidden_dims, dropout=0.3, patience=20, model_path='best_model.pth'):
        super(TabularNN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_dim = X_train.shape[1]
        self.unique_classes = np.unique(y_train)
        self.output_dim = len(self.unique_classes)
        logger.info(f"Unique classes during training: {self.unique_classes}")
        logger.info(f"Output dimension during training: {self.output_dim}")
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.patience = patience
        self.model_path = model_path

        layers = []
        prev_dim = self.input_dim
        for dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.model = nn.Sequential(*layers).to(self.device)

        self.train_loader = self._create_dataloader(X_train, y_train)
        self.val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

    def _create_dataloader(self, X, y, batch_size=64, shuffle=True):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _compute_class_weights(self, y):
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        return torch.tensor(class_weights, dtype=torch.float32).to(self.device)

    def _format_confusion_matrix(self, cm):
        return '\n'.join(['\t'.join(map(str, row)) for row in cm])
    def train_model(self, epochs=50, learning_rate=0.001, batch_size=64):
        class_weights = self._compute_class_weights(self.train_loader.dataset.tensors[1].numpy())
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        best_auc_roc = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0

            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            val_loss, metrics = self.evaluate(criterion)

            current_lr = optimizer.param_groups[0]['lr']

            logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
            logger.info(f"Confusion Matrix:\n{self._format_confusion_matrix(metrics['conf_matrix'])}")

            if metrics['auc_roc'] > best_auc_roc:
                best_auc_roc = metrics['auc_roc']
                patience_counter = 0
                # Save the entire save_dict, not just the state_dict
                save_dict = {
                    'model_state_dict': self.state_dict(),
                    'input_dim': self.input_dim,
                    'hidden_dims': self.hidden_dims,
                    'output_dim': self.output_dim,
                    'target_mapping': {int(k): int(v) for k, v in enumerate(self.unique_classes)}
                }
                torch.save(save_dict, self.model_path)
                logger.info("Best model saved based on highest AUC-ROC.")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping triggered.")
                    break

    def evaluate(self, criterion):
        self.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())
                all_probs.extend(probs)

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_score': f1_score(all_labels, all_preds, average='weighted'),
            'auc_roc': roc_auc_score(all_labels, all_probs, multi_class='ovr'),
            'conf_matrix': confusion_matrix(all_labels, all_preds)
        }

        return val_loss, metrics

    def predict(self, X):
        self.eval()
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

        return preds
    def predict_proba(self, X):
        """Возвращает вероятности классов для входных данных."""
        self.eval()
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

        return probs

    @classmethod
    def load_model(cls, model_path):
        """Загружает модель из файла с восстановлением структуры."""
        checkpoint = torch.load(model_path)

        # Восстанавливаем параметры
        input_dim = checkpoint['input_dim']
        hidden_dims = checkpoint['hidden_dims']
        output_dim = checkpoint['output_dim']
        target_mapping = checkpoint['target_mapping']  # Load target_mapping
        
        logger.info(f"Loading model with input_dim={input_dim}, hidden_dims={hidden_dims}, output_dim={output_dim}")
        logger.info(f"Loaded target mapping: {target_mapping}")

        # Create dummy data with the correct shape
        dummy_X = pd.DataFrame(np.zeros((1, input_dim)))
        dummy_y = pd.Series([0, 1, 2])  # All possible class labels

        # Initialize the model using dummy data
        model = cls(dummy_X, dummy_y, dummy_X, dummy_y, hidden_dims=hidden_dims)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    

"""
1. Что делает последний Linear-слой?
Последний Linear-слой выполняет линейное преобразование входных данных. Он принимает выходы предыдущего слоя (последнего скрытого слоя) и преобразует их в логиты.

Формула для линейного слоя: [ \text{logits} = XW + b ] Где:

(X) — входные данные (выходы предыдущего слоя).
(W) — матрица весов слоя.
(b) — вектор смещений (bias).
(\text{logits}) — выходы слоя (логиты).
Пример: Если у вас 3 класса, то последний Linear-слой будет иметь 3 нейрона, и он выдаст 3 логита для каждого входного примера. Например: [ \text{logits} = [2.5, -1.2, 0.8] ]

2. Что такое логиты?
Логиты — это необработанные выходы модели. Они не интерпретируются как вероятности, потому что:

Они могут быть отрицательными.
Их сумма не равна 1.
Они не ограничены диапазоном [0, 1].
Логиты представляют "уверенность" модели в каждом классе, но в необработанном виде.

3. Зачем нужен softmax?
Функция softmax преобразует логиты в вероятности. Она делает это следующим образом:

Для каждого логита вычисляется экспонента: [ \text{exp}(logit_i) ]
Затем экспоненты нормализуются, чтобы их сумма равнялась 1: [ \text{softmax}(logit_i) = \frac{\text{exp}(logit_i)}{\sum_{j} \text{exp}(logit_j)} ]
Пример: Для логитов ([2.5, -1.2, 0.8]):

Вычисляем экспоненты: [ \text{exp}(2.5) = 12.18, \quad \text{exp}(-1.2) = 0.30, \quad \text{exp}(0.8) = 2.23 ]
Нормализуем: [ \text{softmax}(2.5) = \frac{12.18}{12.18 + 0.30 + 2.23} = 0.83 ] [ \text{softmax}(-1.2) = \frac{0.30}{12.18 + 0.30 + 2.23} = 0.02 ] [ \text{softmax}(0.8) = \frac{2.23}{12.18 + 0.30 + 2.23} = 0.15 ]
Результат: [ \text{softmax}([2.5, -1.2, 0.8]) = [0.83, 0.02, 0.15] ]

Теперь эти значения можно интерпретировать как вероятности принадлежности к каждому классу.

4. Почему softmax не применяется в самом Linear-слое?
softmax не включается в Linear-слой, потому что:

Оптимизация: Функция потерь CrossEntropyLoss в PyTorch ожидает логиты на входе и сама включает softmax в свои вычисления. Это делает обучение более стабильным и эффективным.
Гибкость: Логиты можно использовать для других целей, например, для вычисления рангов или других метрик.
5. Где применяется softmax в коде?
В текущей реализации softmax применяется в следующих методах:

evaluate: Для вычисления вероятностей и метрик (например, AUC-ROC).
predict: Для получения предсказаний классов.
predict_proba: Для получения вероятностей классов.
Пример из метода predict_proba:

Здесь softmax преобразует логиты в вероятности.

6. Итоговое объяснение:
Последний Linear-слой генерирует логиты, которые представляют необработанные выходы модели.
Логиты преобразуются в вероятности с помощью функции softmax, которая нормализует их так, чтобы сумма вероятностей по всем классам равнялась 1.
Это позволяет интерпретировать выходы модели как вероятности принадлежности к каждому классу."
"""