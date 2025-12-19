from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union

from src.infer import SentimentPredictor

app = FastAPI(title="IMDb Sentiment API", version="1.0")

predictor = SentimentPredictor(model_dir="models/distilbert_imdb_small")

class PredictRequest(BaseModel):
    text: Union[str, List[str]]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    # allow either a single string or a list of strings
    if isinstance(req.text, str):
        return predictor.predict(req.text)
    else:
        return [predictor.predict(t) for t in req.text]
