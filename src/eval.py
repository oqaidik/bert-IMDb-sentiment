import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from dataset import IMDbDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_dir = "models/distilbert_imdb_small"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Load dataset and take a random subset from test (to avoid any “easy first chunk” effect)
    ds = load_dataset("imdb")
    test_ds = ds["test"].shuffle(seed=42).select(range(2000))  # 2000 samples

    texts = test_ds["text"]
    labels = test_ds["label"]

    dataset = IMDbDataset(texts, labels, tokenizer, max_length=256)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collator)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, digits=4))

if __name__ == "__main__":
    main()
