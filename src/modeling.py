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
    - Последний Linear-слой генерирует вероятности классов, соответствующие количеству целевых классов.

    4. Стратегия обучения:
    - Используется CrossEntropyLoss с весами классов для учета дисбаланса классов.
    - Модель отслеживает метрику AUC-ROC и сохраняет лучшую модель по её наибольшему значению.
    - Раннее прекращение (early stopping) используется для предотвращения переобучения и экономии вычислительных ресурсов.

    Эта архитектура подходит для табличных данных с несколькими классами и эффективно обрабатывает дисбаланс классов.
    """
    def __init__(self, X_train, y_train, X_val, y_val, hidden_dims, dropout=0.3, patience=5, model_path='best_model.pth'):
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
    
    @classmethod
    def load_model(cls, model_path, X_train, y_train, X_val, y_val):
        """Загружает модель из файла с восстановлением структуры."""
        checkpoint = torch.load(model_path)

        # Восстанавливаем параметры
        input_dim = checkpoint['input_dim']
        hidden_dims = checkpoint['hidden_dims']
        output_dim = checkpoint['output_dim']
        target_mapping = checkpoint['target_mapping']  # Load target_mapping
        
        logger.info(f"Loading model with input_dim={input_dim}, hidden_dims={hidden_dims}, output_dim={output_dim}")
        logger.info(f"Loaded target mapping: {target_mapping}")

        # Инициализируем новую модель
        model = cls(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hidden_dims=hidden_dims
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
        return model