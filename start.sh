#!/bin/bash
# Verificar si PORT está definido (requerido por Render)
if [ -z "$PORT" ]; then
  export PORT=8000  # Valor por defecto para entornos locales
fi

# Forzar modo CPU para evitar errores de CUDA
export CUDA_VISIBLE_DEVICES=""

# Iniciar la aplicación
uvicorn app.main:app --host 0.0.0.0 --port $PORT
