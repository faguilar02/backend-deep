# Usa la imagen específica de Python 3.10.11 con Debian Bullseye (slim)
FROM python:3.10.11-slim-bullseye

# Configura variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Instala dependencias del sistema (necesarias para TensorFlow/PyTorch)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copia e instala dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia toda la aplicación
COPY . .

# Expone el puerto
EXPOSE $PORT

# Comando de inicio
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]