import io
import gc
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import torch
from torchvision import models, transforms
import torch.nn as nn
from pydantic import BaseModel, Field
from typing import Union
import os
import time
import threading
import logging
import traceback
import psutil

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Deshabilita cualquier GPU
device = torch.device("cpu")  # Forzar PyTorch a usar solo CPU

# Definir la arquitectura del modelo de regresión
def create_resnet_model():
    logger.info("Creando arquitectura ResNet...")
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 1)
    )
    return model

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Banana Ripeness Predictor",
    description="API para predecir madurez de plátanos usando modelos de DL",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todos los headers
)

# Variables globales para los modelos y estado de carga
classifier = None
model_reg = None
models_loaded = False
loading_in_progress = False
load_error = None
load_lock = threading.Lock()

def load_classifier_model(path):
    """Carga el modelo de clasificación con manejo explícito de memoria"""
    logger.info(f"Cargando modelo de clasificación desde: {path}")
    
    # Liberar memoria antes de cargar
    tf.keras.backend.clear_session()
    tf.config.set_visible_devices([], 'GPU')  # Forzar CPU
    
    # Configurar TensorFlow para bajo consumo de memoria
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # Estrategia de carga segura
    try:
        logger.info("Configurando carga optimizada de Keras...")
        model = tf.keras.models.load_model(
            path,
            compile=False  # No compilar para ahorrar memoria
        )
        logger.info("✅ Modelo de clasificación cargado en CPU")
        return model
    except Exception as e:
        logger.error(f"❌ Error cargando modelo de clasificación: {e}")
        logger.error(traceback.format_exc())
        return None
    finally:
        # Limpiar cualquier recurso residual
        gc.collect()

def load_regression_model(path):
    """Carga el modelo de regresión con manejo explícito de memoria"""
    logger.info(f"Cargando modelo de regresión desde: {path}")
    
    try:
        # Crear arquitectura
        logger.info("Instanciando modelo ResNet...")
        model = create_resnet_model().to(device)
        
        # Cargar pesos
        logger.info("Cargando pesos del modelo...")
        state_dict = torch.load(
            path, 
            map_location=device,
            weights_only=True  # Modo seguro
        )
        
        # Ajustar claves si es necesario
        logger.info("Ajustando claves del state_dict...")
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Cargar pesos en el modelo
        logger.info("Cargando pesos en la arquitectura...")
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        logger.info("✅ Modelo de regresión cargado correctamente")
        return model
    except Exception as e:
        logger.error(f"❌ Error cargando modelo de regresión: {e}")
        logger.error(traceback.format_exc())
        return None
    finally:
        # Limpiar memoria
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def log_memory_usage():
    """Registra el uso actual de memoria"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Uso de memoria: {mem_info.rss / 1024 / 1024:.2f} MB")

def load_models():
    global classifier, model_reg, models_loaded, loading_in_progress, load_error
    
    # Si ya están cargados o se está cargando, salir
    if models_loaded or loading_in_progress:
        return
    
    with load_lock:
        if models_loaded or loading_in_progress:
            return
            
        loading_in_progress = True
        load_error = None
        start_time = time.time()
        logger.info("⚡ Cargando modelos bajo demanda...")
        log_memory_usage()
        
        try:
            # Construir rutas a los modelos
            model_dir = os.path.join(os.path.dirname(__file__), "models")
            classifier_path = os.path.join(model_dir, "clasificacion_model_v4.h5")
            regression_path = os.path.join(model_dir, "best_banana_ripeness_regression.pth")
            
            # Verificar existencia de archivos
            if not os.path.exists(classifier_path):
                raise FileNotFoundError(f"Archivo no encontrado: {classifier_path}")
            if not os.path.exists(regression_path):
                raise FileNotFoundError(f"Archivo no encontrado: {regression_path}")
            
            logger.info(f"Tamaño modelo clasificación: {os.path.getsize(classifier_path) / 1024 / 1024:.2f} MB")
            logger.info(f"Tamaño modelo regresión: {os.path.getsize(regression_path) / 1024 / 1024:.2f} MB")
            log_memory_usage()
            
            # Cargar modelos en secuencia para ahorrar memoria
            logger.info("Cargando modelo de clasificación...")
            classifier = load_classifier_model(classifier_path)
            log_memory_usage()
            
            if classifier is None:
                raise RuntimeError("Fallo al cargar modelo de clasificación")
            
            # Liberar recursos antes de cargar el segundo modelo
            time.sleep(0.5)
            gc.collect()
            log_memory_usage()
            
            logger.info("Cargando modelo de regresión...")
            model_reg = load_regression_model(regression_path)
            log_memory_usage()
            
            if model_reg is None:
                raise RuntimeError("Fallo al cargar modelo de regresión")
            
            # Marcar como cargados
            models_loaded = True
            load_time = time.time() - start_time
            logger.info(f"⏱️ Tiempo total de carga: {load_time:.2f} segundos")
            
        except Exception as e:
            load_error = str(e)
            logger.error(f"❌❌❌ ERROR CRÍTICO: {load_error}")
            logger.error(traceback.format_exc())
            
        finally:
            loading_in_progress = False
            log_memory_usage()

# Endpoint raíz
@app.get("/")
async def root():
    return {"message": "Banana Ripeness Predictor API"}

# Endpoint de live para Render
@app.get("/live")
async def liveness_check():
    return {"status": "alive", "models_loaded": models_loaded}

# Transformaciones para PyTorch
transform_reg = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Modelo de respuesta
class PredictionResult(BaseModel):
    is_banana: bool
    days_remaining: Union[float, None] = Field(None)
    message: str

# Endpoint de predicción (con carga diferida)
@app.post("/predict", response_model=PredictionResult)
async def predict_banana(file: UploadFile = File(...)):
    global models_loaded, load_error
    
    # Cargar modelos si aún no están cargados
    if not models_loaded:
        logger.info("Solicitud de predicción activa carga de modelos...")
        load_models()
    
    # Manejar errores de carga
    if not models_loaded:
        error_detail = "Los modelos no están cargados correctamente."
        if load_error:
            error_detail += f" Error: {load_error}"
        
        raise HTTPException(
            status_code=503,
            detail=error_detail
        )
    
    # Verificar tipo de archivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Formato de archivo no válido. Por favor sube una imagen (jpeg, png, etc.)"
        )
    
    try:
        # Leer la imagen
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        logger.info(f"Imagen recibida: {file.filename}, tamaño: {img.size}")
        
        # --- Clasificación ---
        # Preprocesar imagen para TensorFlow
        img_tf = tf.keras.preprocessing.image.img_to_array(img)
        img_tf = tf.image.resize(img_tf, [224, 224])
        img_tf = (img_tf / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_tf = tf.expand_dims(img_tf, axis=0)
        
        # Predecir con TensorFlow en CPU
        clf_pred = classifier.predict(img_tf, verbose=0)[0][0]
        banana_prob = 1 - clf_pred
        logger.info(f"Probabilidad de ser plátano: {banana_prob:.4f}")
        
        # Si no es plátano
        if banana_prob < 0.5:
            return PredictionResult(
                is_banana=False,
                days_remaining=None,
                message="No es un plátano"
            )
        
        # --- Regresión ---
        # Preprocesar imagen para PyTorch
        img_reg = transform_reg(img)
        img_reg = img_reg.unsqueeze(0).to(device)
        
        # Predecir días restantes
        with torch.no_grad():
            days = max(0, round(model_reg(img_reg).item(), 1))
        
        return PredictionResult(
            is_banana=True,
            days_remaining=days,
            message=f"Plátano detectado. Días restantes: {days:.1f}"
        )
        
    except Exception as e:
        logger.error(f"Error procesando imagen: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la imagen: {str(e)}"
        )
    finally:
        # Limpiar memoria después de la predicción
        gc.collect()

# Endpoint de verificación de salud
@app.get("/health")
async def health_check():
    return {
        "status": "OK" if models_loaded else "LOADING",
        "models_loaded": models_loaded,
        "loading_in_progress": loading_in_progress,
        "load_error": load_error if not models_loaded else None,
        "device": str(device)
    }

@app.get("/force-load")
async def force_load_models():
    """Endpoint para forzar la carga de modelos manualmente"""
    if models_loaded:
        return {"status": "already_loaded"}
    
    logger.info("Forzando carga de modelos...")
    load_models()
    return {
        "status": "loading_triggered",
        "models_loaded": models_loaded,
        "error": load_error
    }

@app.on_event("startup")
async def startup_event():
    logger.info("Aplicación iniciada. Los modelos se cargarán bajo demanda.")