import os
import torch
from torch.quantization import quantize_dynamic
from torchvision import models
import torch.nn as nn

# Obtener ruta absoluta
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_banana_ripeness_regression.pth")
OUTPUT_PATH = os.path.join(MODEL_DIR, "best_banana_ripeness_regression_quantized.pt")

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

# Cargar modelo
model = create_resnet_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Cuantizar el modelo
quantized_model = quantize_dynamic(
    model,
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)

# Exportar a TorchScript
scripted_model = torch.jit.script(quantized_model)
scripted_model.save(OUTPUT_PATH)

print(f"✅ Modelo cuantizado exportado a TorchScript: {OUTPUT_PATH}")