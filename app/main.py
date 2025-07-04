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
from typing import Union, Optional
import os
import time
import threading
import logging
import traceback
import psutil

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Forzar uso de CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
device = torch.device("cpu")
torch.set_num_threads(1)

# Inicializar FastAPI
app = FastAPI(
    title="Banana Ripeness Predictor",
    description="API optimizada para predecir madurez de plátanos",
    version="3.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
classifier: Optional[tf.lite.Interpreter] = None
model_reg: Optional[torch.nn.Module] = None
models_loaded = False
loading_in_progress = False
load_error: Optional[str] = None
load_lock = threading.Lock()

# Transformaciones para PyTorch
transform_reg = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Uso de memoria: {mem_info.rss / 1024 / 1024:.2f} MB")

def create_resnet_model():
    """Crea arquitectura base sin cuantizar"""
    logger.info("Creando arquitectura ResNet base...")
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

def load_classifier_model(path: str):
    """Carga modelo TFLite optimizado"""
    logger.info(f"Cargando modelo TFLite desde: {path}")
    try:
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        logger.info("✅ Modelo clasificador TFLite cargado")
        return interpreter
    except Exception as e:
        logger.error(f"❌ Error cargando TFLite: {e}")
        return None
    finally:
        gc.collect()

def load_regression_model(path: str):
    """Carga modelo de regresión (cuantizado o normal)"""
    logger.info(f"Cargando modelo desde: {path}")
    try:
        # Primero intenta cargar como modelo cuantizado completo
        try:
            model = torch.jit.load(path, map_location=device)
            model.eval()
            logger.info("✅ Modelo cuantizado (TorchScript) cargado")
            return model
        except:
            # Si falla, intenta cargar como modelo normal
            model = create_resnet_model().to(device)
            state_dict = torch.load(path, map_location=device)
            
            # Manejo de nombres de parámetros
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
            logger.info("✅ Modelo normal cargado")
            return model
            
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")
        logger.error(traceback.format_exc())
        return None
    finally:
        gc.collect()

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
        logger.info("⚡ Iniciando carga optimizada de modelos...")
        log_memory_usage()
        
        try:
            model_dir = os.path.join(os.path.dirname(__file__), "models")
            
            # Cargar clasificador TFLite
            classifier_path = os.path.join(model_dir, "clasificacion_model_v4_opt.tflite")
            if not os.path.exists(classifier_path):
                raise FileNotFoundError(f"Modelo TFLite no encontrado: {classifier_path}")
            
            logger.info("Cargando clasificador optimizado...")
            classifier = load_classifier_model(classifier_path)
            if not classifier:
                raise RuntimeError("Fallo al cargar clasificador TFLite")
            
            time.sleep(2)
            gc.collect()
            log_memory_usage()
            
            # Cargar modelo de regresión - usa el modelo original como respaldo
            regression_path = os.path.join(model_dir, "best_banana_ripeness_regression.pth")
            if not os.path.exists(regression_path):
                # Intenta con el cuantizado si existe
                quantized_path = os.path.join(model_dir, "best_banana_ripeness_regression_quantized.pth")
                if os.path.exists(quantized_path):
                    regression_path = quantized_path
                else:
                    raise FileNotFoundError("No se encontró modelo de regresión")
            
            logger.info(f"Cargando modelo de regresión: {regression_path}")
            model_reg = load_regression_model(regression_path)
            if not model_reg:
                raise RuntimeError("Fallo al cargar modelo de regresión")
            
            models_loaded = True
            logger.info(f"✅✅ Modelos cargados en {time.time() - start_time:.2f} segundos")
            
        except Exception as e:
            load_error = str(e)
            logger.error(f"❌ ERROR: {load_error}")
            logger.error(traceback.format_exc())
        finally:
            loading_in_progress = False
            log_memory_usage()

# Endpoints
@app.get("/")
async def root():
    return {"message": "Banana Ripeness Predictor API (Optimizada)"}

@app.get("/live")
async def liveness_check():
    return {"status": "alive", "models_loaded": models_loaded}

class PredictionResult(BaseModel):
    is_banana: bool
    days_remaining: Optional[float] = Field(None)
    message: str

@app.post("/predict", response_model=PredictionResult)
async def predict_banana(file: UploadFile = File(...)):
    global models_loaded, load_error
    
    if not models_loaded:
        load_models()
    
    if not models_loaded:
        error_msg = "Modelos no cargados"
        if load_error:
            error_msg += f": {load_error}"
        raise HTTPException(status_code=503, detail=error_msg)
    
    try:
        # Leer y procesar imagen
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        logger.info(f"Imagen recibida: {file.filename}, tamaño: {img.size}")
        
        # --- Clasificación con TFLite ---
        img_tf = np.array(img.resize((224, 224)), dtype=np.float32)
        img_tf = img_tf / 255.0
        
        input_details = classifier.get_input_details()
        output_details = classifier.get_output_details()
        
        classifier.set_tensor(input_details[0]['index'], np.expand_dims(img_tf, axis=0))
        classifier.invoke()
        
        clf_pred = classifier.get_tensor(output_details[0]['index'])[0][0]
        banana_prob = 1 - clf_pred
        logger.info(f"Probabilidad de ser plátano: {banana_prob:.4f}")
        
        if banana_prob < 0.5:
            return PredictionResult(
                is_banana=False,
                message="No es un plátano"
            )
        
        # --- Regresión ---
        img_reg = transform_reg(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if isinstance(model_reg, torch.jit.ScriptModule):
                # Modelo TorchScript (cuantizado)
                output = model_reg(img_reg)
                days = output[0].item()
            else:
                # Modelo normal
                days = model_reg(img_reg).item()
            
            days = max(0, round(days, 1))
        
        return PredictionResult(
            is_banana=True,
            days_remaining=days,
            message=f"Plátano detectado. Días restantes: {days:.1f}"
        )
        
    except Exception as e:
        logger.error(f"Error procesando imagen: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {str(e)}")
    finally:
        gc.collect()

@app.get("/preload")
async def preload_models():
    """Endpoint para precargar modelos manualmente después del despliegue"""
    if models_loaded:
        return {"status": "already_loaded"}
    
    thread = threading.Thread(target=load_models)
    thread.start()
    return {"status": "loading_started"}

@app.get("/health")
async def health_check():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        "status": "OK" if models_loaded else "LOADING",
        "memory_usage_mb": f"{mem_info.rss / 1024 / 1024:.2f}",
        "models_loaded": models_loaded,
        "load_error": load_error
    }

@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando con carga bajo demanda...")