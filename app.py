import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union

USE_ONNX = os.getenv("USE_ONNX", "1") == "1"
DUMMY_MODEL = os.getenv("DUMMY_MODEL", "0") == "1"

app = FastAPI(title="IMDb Sentiment API", version="1.0")

class PredictRequest(BaseModel):
    text: Union[str, List[str]]

predictor = None

@app.on_event("startup")
def load_predictor():
    global predictor
    if DUMMY_MODEL:
        predictor = _DummyPredictor()
        return

    if USE_ONNX:
        from src.infer_onnx import ONNXSentimentPredictor
        predictor = ONNXSentimentPredictor()
    else:
        from src.infer import SentimentPredictor
        predictor = SentimentPredictor(model_dir="models/distilbert_imdb_small")

@app.get("/health")
def health():
    return {"status": "ok", "backend": "dummy" if DUMMY_MODEL else ("onnx" if USE_ONNX else "torch")}

@app.post("/predict")
def predict(req: PredictRequest):
    if predictor is None:
        return {"error": "Predictor not loaded"}

    if isinstance(req.text, str):
        return predictor.predict(req.text)
    else:
        return [predictor.predict(t) for t in req.text]

class _DummyPredictor:
    def predict(self, text: str):
        return {
            "label": "positive",
            "confidence": 0.5,
            "probabilities": {"negative": 0.5, "positive": 0.5},
        }
