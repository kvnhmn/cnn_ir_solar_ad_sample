import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from serving.inference import load_model, predict
from core.preprocess import preprocess

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "models/irnet.pt")
DEVICE = os.getenv("DEVICE", "cpu")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"] = load_model(WEIGHTS_PATH, DEVICE)
    _state["device"] = DEVICE
    yield
    _state.clear()


app = FastAPI(
    title="IR Solar Module Anomaly Classifier",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    label: int
    class_name: str
    p_anomaly: float
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Image could not be read.")

    tensor = preprocess(image)
    return predict(_state["model"], tensor, _state["device"])