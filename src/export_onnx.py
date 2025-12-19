import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    model_dir = "models/distilbert_imdb_small"
    onnx_dir = os.path.join("models", "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "distilbert_imdb.onnx")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    # Dummy input
    text = "This movie was amazing."
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Export
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    print(f"✅ Exported ONNX model to: {onnx_path}")

if __name__ == "__main__":
    main()
