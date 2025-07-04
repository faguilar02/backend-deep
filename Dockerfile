FROM python:3.10.11-slim-bullseye

# Configura variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copia e instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia la aplicación
COPY . .

# Verificación explícita de los modelos
RUN echo "=== Verificando la presencia de los modelos ===" && \
    ls -lR /app/app/models

# Solución: Esperar 25 segundos antes de iniciar para evitar el escaneo temprano de Render
CMD ["sh", "-c", "echo 'PORT variable: $PORT' && \
     echo '=== Verificando modelos ===' && \
     ls -l /app/app/models && \
     echo '=== Esperando 25 segundos para iniciar ===' && \
     sleep 25 && \
     echo '=== Iniciando Uvicorn ===' && \
     uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]