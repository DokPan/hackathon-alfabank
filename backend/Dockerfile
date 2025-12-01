FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p src/models

COPY src/ ./src/
COPY *.csv ./  # Копируем CSV файлы с данными

RUN ln -sf /app/src /src

EXPOSE 8000

