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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Deshabilita cualquier GPU
device = torch.device("cpu")  # Forzar PyTorch a usar solo CPU

# Definir la arquitectura del modelo de regresión
def create_resnet_model():
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

# Endpoint raíz para verificación de inicio (CRÍTICO PARA RENDER)
@app.get("/")
async def root():
    return {"message": "Banana Ripeness Predictor API is initializing. Please wait."}

# Variables globales para los modelos
classifier = None
model_reg = None

# Cargar modelos al iniciar
@app.on_event("startup")
async def startup_event():
    print("⚡ Evento startup iniciado")
async def load_models():
    global classifier, model_reg
    
    # Limpiar sesión de TensorFlow
    tf.keras.backend.clear_session()

    # Forzar TensorFlow a usar CPU
    tf.config.set_visible_devices([], 'GPU')
    
    # Construir rutas a los modelos
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    classifier_path = os.path.join(model_dir, "clasificacion_model_v4.h5")
    regression_path = os.path.join(model_dir, "best_banana_ripeness_regression.pth")
    
    print(f"Intentando cargar modelo de clasificación desde: {classifier_path}")
    print(f"Intentando cargar modelo de regresión desde: {regression_path}")
    
    # Cargar modelo de clasificación en CPU
    try:
        classifier = tf.keras.models.load_model(classifier_path)
        print("✅ Modelo de clasificación cargado en CPU")
    except Exception as e:
        print(f"❌ Error cargando modelo de clasificación: {e}")
        classifier = None
    
    # Cargar modelo de regresión
    try:
        model_reg = create_resnet_model().to(device)
        state_dict = torch.load(regression_path, map_location=device)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model_reg.load_state_dict(new_state_dict, strict=False)
        model_reg.eval()
        print("✅ Modelo de regresión cargado correctamente")
    except Exception as e:
        print(f"❌ Error cargando modelo de regresión: {e}")
        model_reg = None

    # AGREGAR ESTO AL FINAL
    print("="*50)
    print(f"✅ Clasificador cargado: {classifier is not None}")
    print(f"✅ Modelo regresión cargado: {model_reg is not None}")
    print("="*50)
    print("🟢 TODOS LOS MODELOS CARGADOS - APLICACIÓN LISTA")

# Transformaciones para PyTorch
transform_reg = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Modelo de respuesta actualizado
class PredictionResult(BaseModel):
    is_banana: bool
    days_remaining: Union[float, None] = Field(None)
    message: str

# Endpoint de predicción
@app.post("/predict", response_model=PredictionResult)
async def predict_banana(file: UploadFile = File(...)):
    # Verificar que los modelos estén cargados
    if classifier is None or model_reg is None:
        raise HTTPException(
            status_code=503,
            detail="Los modelos no están cargados correctamente. Intente más tarde."
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
        print(f"Imagen recibida: {file.filename}, tamaño: {img.size}")
        
        # --- Clasificación ---
        # Preprocesar imagen para TensorFlow
        img_tf = tf.keras.preprocessing.image.img_to_array(img)
        img_tf = tf.image.resize(img_tf, [224, 224])
        img_tf = (img_tf / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_tf = tf.expand_dims(img_tf, axis=0)
        
        # Predecir con TensorFlow en CPU
        clf_pred = classifier.predict(img_tf, verbose=0)[0][0]
        banana_prob = 1 - clf_pred
        print(f"Probabilidad de ser plátano: {banana_prob:.4f}")
        
        # Si no es plátano
        if banana_prob < 0.5:
            return PredictionResult(
                is_banana=False,
                days_remaining=None,
                message="No es un plátano"
            )
        
        # --- Regresión ---
        # Limpiar memoria
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la imagen: {str(e)}"
        )

# Endpoint de verificación de salud
@app.get("/health")
async def health_check():
    models_loaded = classifier is not None and model_reg is not None
    return {
        "status": "OK" if models_loaded else "ERROR",
        "models_loaded": models_loaded,
        "device": str(device)
    }


@app.get("/test")
async def test_endpoint():
    return {
        "status": "OK",
        "port_info": f"La aplicación debería estar en el puerto {os.getenv('PORT', 8000)}"
    }