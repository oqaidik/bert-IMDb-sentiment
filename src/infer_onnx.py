import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

class ONNXSentimentPredictor:
    def __init__(self, model_dir="models/distilbert_imdb_small", onnx_path="models/onnx/distilbert_imdb.onnx"):
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(
                f"ONNX file not found: {onnx_path}\n"
                "Run: python src/export_onnx.py"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

        # CPUExecutionProvider is default; explicit for clarity
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def predict(self, text: str, max_length: int = 128):
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="np"
        )

        inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }

        logits = self.session.run(["logits"], inputs)[0]  # shape (batch, 2)
        probs = softmax(logits[0])  # first item in batch

        pred = int(np.argmax(probs))
        label = "positive" if pred == 1 else "negative"

        return {
            "label": label,
            "confidence": float(probs[pred]),
            "probabilities": {"negative": float(probs[0]), "positive": float(probs[1])},
        }

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)
