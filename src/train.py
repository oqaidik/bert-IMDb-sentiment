import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

# Your Dataset class
from dataset import IMDbDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- Config (small subset for learning) ---
    model_name = "distilbert-base-uncased"
    max_length = 256

    train_subset_size = 2000   # small for fast iteration
    test_subset_size = 1000

    batch_size = 16
    lr = 2e-5
    epochs = 2

    output_dir = "models/distilbert_imdb_small"
    os.makedirs(output_dir, exist_ok=True)

    # --- Load IMDb dataset (HuggingFace datasets) ---
    ds = load_dataset("imdb")

    # Shuffle then take subset (IMPORTANT to avoid label-ordered slices)
    train_small = ds["train"].shuffle(seed=42).select(range(train_subset_size))
    test_small  = ds["test"].shuffle(seed=42).select(range(test_subset_size))

    # Small subset for learning
    train_texts  = train_small["text"]
    train_labels = train_small["label"]

    test_texts  = test_small["text"]
    test_labels = test_small["label"]

    print("Train label counts:", np.bincount(train_labels))
    print("Test  label counts:", np.bincount(test_labels))

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- PyTorch Dataset ---
    train_dataset = IMDbDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    test_dataset = IMDbDataset(test_texts, test_labels, tokenizer, max_length=max_length)

    # --- Data collator (dynamic padding per batch) ---
    # Important: our Dataset already pads to max_length.
    # Using a collator is still fine; it keeps batching consistent.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    # --- Model ---
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # --- Training loop ---
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward (loss is computed internally because we pass labels)
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if step % 50 == 0:
                avg_loss = running_loss / step
                print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | Avg Loss: {avg_loss:.4f}")

    # --- Quick evaluation (accuracy only for now) ---
    # --- Quick evaluation (detect collapse to one class) ---
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    p0 = (all_preds == 0).mean()
    p1 = (all_preds == 1).mean()
    print(f"[Quick Eval] Acc: {acc:.4f} | Pred% class0: {p0:.3f} | Pred% class1: {p1:.3f}")

    # --- Save model + tokenizer ---
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Saved to:", output_dir)



if __name__ == "__main__":
    main()
