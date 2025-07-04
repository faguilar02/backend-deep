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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Reduce logs de TensorFlow
device = torch.device("cpu")  # Forzar PyTorch a usar solo CPU

# Limitar threads para reducir consumo de memoria
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
torch.set_num_threads(1)

# Definir la arquitectura del modelo de regresión
def create_resnet_model():
    logger.info("Creando arquitectura ResNet optimizada...")
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    
    # Reducir tamaño de capas para ahorrar memoria
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),  # Reducido de 512
        nn.ReLU(),
        nn.Dropout(0.1),           # Reducido de 0.2
        nn.Linear(256, 64),         # Reducido de 128
        nn.ReLU(),
        nn.Dropout(0.05),          # Reducido de 0.1
        nn.Linear(64, 1)
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para los modelos y estado de carga
classifier = None
model_reg = None
models_loaded = False
loading_in_progress = False
load_error = None
load_lock = threading.Lock()

def load_classifier_model(path):
    """Carga el modelo de clasificación optimizado para memoria"""
    logger.info(f"Cargando modelo de clasificación optimizado desde: {path}")
    
    try:
        # Liberar memoria antes de cargar
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Cargar con optimizaciones de memoria
        model = tf.keras.models.load_model(
            path,
            compile=False
        )
        
        # Compresión adicional del modelo
        model = tf.keras.models.clone_model(model)
        model.set_weights(model.get_weights())
        
        logger.info("✅ Modelo de clasificación cargado (optimizado)")
        return model
    except Exception as e:
        logger.error(f"❌ Error cargando modelo de clasificación: {e}")
        logger.error(traceback.format_exc())
        return None
    finally:
        gc.collect()

def load_regression_model(path):
    """Carga el modelo de regresión con cuantización para ahorrar memoria"""
    logger.info(f"Cargando modelo de regresión cuantizado desde: {path}")
    
    try:
        # Crear arquitectura optimizada
        model = create_resnet_model().to(device)
        
        # Cargar pesos con modo seguro
        state_dict = torch.load(
            path, 
            map_location=device,
            weights_only=True
        )
        
        # Aplicar cuantización dinámica (reduce memoria ~4x)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        # Cargar pesos en el modelo cuantizado
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        quantized_model.load_state_dict(state_dict, strict=False)
        quantized_model.eval()
        
        logger.info("✅ Modelo de regresión cuantizado cargado")
        return quantized_model
    except Exception as e:
        logger.error(f"❌ Error cargando modelo de regresión: {e}")
        logger.error(traceback.format_exc())
        return None
    finally:
        gc.collect()
        if 'torch' in globals():
            torch.cuda.empty_cache()

def log_memory_usage():
    """Registra el uso actual de memoria"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Uso de memoria: {mem_info.rss / 1024 / 1024:.2f} MB")

def load_models():
    global classifier, model_reg, models_loaded, loading_in_progress, load_error
    
    if models_loaded or loading_in_progress:
        return
    
    with load_lock:
        if models_loaded or loading_in_progress:
            return
            
        loading_in_progress = True
        load_error = None
        start_time = time.time()
        logger.info("⚡ Cargando modelos con optimización de memoria...")
        log_memory_usage()
        
        try:
            model_dir = os.path.join(os.path.dirname(__file__), "models")
            classifier_path = os.path.join(model_dir, "clasificacion_model_v4.h5")
            regression_path = os.path.join(model_dir, "best_banana_ripeness_regression.pth")
            
            # Carga secuencial con limpieza entre modelos
            logger.info("Cargando modelo de clasificación...")
            classifier = load_classifier_model(classifier_path)
            log_memory_usage()
            
            if classifier is None:
                raise RuntimeError("Fallo al cargar modelo de clasificación")
            
            # Limpieza agresiva entre cargas
            time.sleep(0.5)
            gc.collect()
            tf.keras.backend.clear_session()
            
            logger.info("Cargando modelo de regresión...")
            model_reg = load_regression_model(regression_path)
            log_memory_usage()
            
            if model_reg is None:
                raise RuntimeError("Fallo al cargar modelo de regresión")
            
            models_loaded = True
            load_time = time.time() - start_time
            logger.info(f"⏱️ Tiempo total de carga: {load_time:.2f} segundos")
            
        except Exception as e:
            load_error = str(e)
            logger.error(f"❌❌❌ ERROR: {load_error}")
            logger.error(traceback.format_exc())
            
        finally:
            loading_in_progress = False
            log_memory_usage()

# Endpoint raíz
@app.get("/")
async def root():
    return {"message": "Banana Ripeness Predictor API"}

@app.get("/live")
async def liveness_check():
    return {"status": "alive", "models_loaded": models_loaded}

# Transformaciones optimizadas
transform_reg = transforms.Compose([
    transforms.Resize(128),  # Reducido de 256
    transforms.CenterCrop(112),  # Reducido de 224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Modelo de respuesta
class PredictionResult(BaseModel):
    is_banana: bool
    days_remaining: Union[float, None] = Field(None)
    message: str

@app.post("/predict", response_model=PredictionResult)
async def predict_banana(file: UploadFile = File(...)):
    global models_loaded, load_error
    
    if not models_loaded:
        logger.info("Iniciando carga bajo demanda...")
        load_models()
    
    if not models_loaded:
        error_detail = "Error cargando modelos"
        if load_error:
            error_detail += f": {load_error}"
        raise HTTPException(status_code=503, detail=error_detail)
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Formato de archivo no válido")
    
    try:
        # Leer y redimensionar imagen inmediatamente para ahorrar memoria
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize((224, 224))  # Redimensionar temprano
        
        # --- Clasificación ---
        img_tf = tf.keras.preprocessing.image.img_to_array(img)
        img_tf = tf.image.resize(img_tf, [224, 224])
        img_tf = (img_tf / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_tf = tf.expand_dims(img_tf, axis=0)
        
        # Predecir en lotes de 1
        clf_pred = classifier.predict(img_tf, verbose=0, batch_size=1)[0][0]
        banana_prob = 1 - clf_pred
        logger.info(f"Probabilidad de ser plátano: {banana_prob:.4f}")
        
        if banana_prob < 0.5:
            return PredictionResult(
                is_banana=False,
                days_remaining=None,
                message="No es un plátano"
            )
        
        # --- Regresión ---
        img_reg = transform_reg(img)
        img_reg = img_reg.unsqueeze(0).to(device)
        
        with torch.no_grad():
            days = max(0, round(model_reg(img_reg).item(), 1))
        
        return PredictionResult(
            is_banana=True,
            days_remaining=days,
            message=f"Plátano detectado. Días restantes: {days:.1f}"
        )
        
    except Exception as e:
        logger.error(f"Error procesando imagen: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen")
    finally:
        gc.collect()

@app.get("/health")
async def health_check():
    return {
        "status": "OK" if models_loaded else "LOADING",
        "models_loaded": models_loaded,
        "memory_usage": f"{psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB"
    }

@app.get("/force-load")
async def force_load_models():
    if models_loaded:
        return {"status": "already_loaded"}
    
    logger.info("Forzando carga optimizada de modelos...")
    load_models()
    return {"status": "loading_triggered"}

@app.on_event("startup")
async def startup_event():
    logger.info("Aplicación iniciada - Modo de memoria optimizada")