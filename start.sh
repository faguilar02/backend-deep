#!/bin/bash
# Verificar si PORT está definido (requerido por Render)
if [ -z "$PORT" ]; then
  export PORT=8000  # Valor por defecto para entornos locales
fi

# Iniciar la aplicación
uvicorn app.main:app --host 0.0.0.0 --port $PORT