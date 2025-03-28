# Используем базовый образ Python
FROM python:3.12-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копирование и установка зависимостей
COPY requirements-docker.txt requirements-docker.txt
RUN pip install --upgrade pip && \
    pip install -r requirements-docker.txt

# Копирование проекта
COPY src/ /app/src/
COPY main.py /app/

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements-docker.txt

# Открываем порт
EXPOSE 8000

# Запуск FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
