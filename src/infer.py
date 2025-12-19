import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentPredictor:
    def __init__(self, model_dir="models/distilbert_imdb_small", device=None):
        # CPU optimization knobs
        torch.set_num_threads(max(1, os.cpu_count() // 2))  # use half cores (often faster)
        torch.set_float32_matmul_precision("high")  # harmless on CPU, helps on some builds

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, max_length: int = 128):
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.inference_mode():
            out = self.model(**enc)
            probs = torch.softmax(out.logits, dim=1).squeeze(0)

        pred = int(torch.argmax(probs).item())
        label = "positive" if pred == 1 else "negative"

        return {
            "label": label,
            "confidence": float(probs[pred].item()),
            "probabilities": {
                "negative": float(probs[0].item()),
                "positive": float(probs[1].item()),
            },
        }
