# FastAPI
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

# Tensorflow
from tensorflow.keras.models import load_model
import numpy as np
import json

# Recursos de prediccion
from app.util.image import coat, labels, a_boot

# Modelo
from app.models.request import Predict

# Modelo
modelo = load_model("app\kerasm\model.keras")

# Router
router = APIRouter(prefix="/predict")


# Controladores de rutas
@router.get("/saludo", response_class=HTMLResponse)
def saludo():
    return "<h1>Hi Tensorflow!</h1>"


@router.get("/image", response_class=JSONResponse)
def predecir():
    # Arreglo numpy de Coat
    img = np.array(coat, dtype=np.float32)
    img_input = img.reshape(1, 28, 28, 1)
    pred = modelo.predict(img_input)
    element = labels[np.argmax(pred)]
    return {
        "Status": "success",
        "Imágen": element,
    }


@router.post("/newImage", response_class=JSONResponse)
def predict(predict: Predict):

    # Prediccion
    if predict.model == "clothes":
        # Procesamiento de imágen
        img = predict.image
        image_np = np.array(img, dtype=np.float32)
        image_np = image_np.reshape(1, 28, 28, 1)
        pred = modelo.predict(image_np)
        element = labels[np.argmax(pred)]
        return {
            "Status": "success",
            "Imágen": element,
        }
    else:
        return {"Status": "Not found!"}
