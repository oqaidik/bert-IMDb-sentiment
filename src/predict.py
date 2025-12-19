import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def predict(text: str, model_dir: str, device: str):
    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()

    enc = tokenizer(
        text,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.inference_mode():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=1).squeeze(0)
        pred = torch.argmax(probs).item()

    label = "positive" if pred == 1 else "negative"
    confidence = probs[pred].item()

    return label, confidence, probs.cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    parser.add_argument("--model_dir", type=str, default="models/distilbert_imdb_small")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    label, conf, probs = predict(args.text, args.model_dir, args.device)

    print(f"Prediction: {label} (confidence={conf:.4f})")
    print(f"Probabilities [neg, pos]: {probs}")


if __name__ == "__main__":
    main()
