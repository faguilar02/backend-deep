FROM python:3.10.11-slim-bullseye

# Configura variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 
# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instalar psutil para monitoreo de memoria
RUN pip install --no-cache-dir psutil

# Directorio de trabajo
WORKDIR /app

# Copia solo requirements.txt primero para cachear dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia la aplicación
COPY . .

# Iniciar Uvicorn con optimización de memoria
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --limit-max-requests 20 --timeout-keep-alive 15"]