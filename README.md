# IMDb Sentiment Analysis API with DistilBERT

This project implements an **end-to-end NLP pipeline** for sentiment analysis using a **pretrained Transformer (DistilBERT)** fine-tuned on the IMDb movie reviews dataset.

It covers the **full machine learning lifecycle**:
- dataset preparation
- fine-tuning a Transformer (transfer learning)
- robust evaluation (Accuracy, F1, confusion matrix)
- inference on new text
- deployment as a REST API using FastAPI

The project is designed as a **portfolio-grade example** of modern NLP engineering.

---

## 🎯 Task Description

- **Task**: Binary sentiment classification  
- **Labels**:
  - `0` → Negative
  - `1` → Positive
- **Input**: Raw movie review text
- **Output**: Sentiment label + confidence score

---

## 🧠 Model & Dataset

### Model
- **DistilBERT** (`distilbert-base-uncased`)
- Lightweight Transformer optimized for speed
- Fine-tuned using supervised learning

### Dataset
- **IMDb Movie Reviews**
- Source: Hugging Face `datasets`
- 50,000 labeled reviews
- Balanced positive / negative classes

---

## ⚙️ Training Setup

- Transfer learning (fine-tuning pretrained Transformer)
- Dynamic padding using `DataCollatorWithPadding`
- Optimizer: AdamW
- Learning rate: `2e-5`
- Max sequence length: `256` (training), `128` (inference)
- Training performed on a **balanced shuffled subset** for fast iteration

---

## 📊 Evaluation Results

Evaluation on a shuffled test subset (2,000 samples):

| Metric | Score |
|------|------|
| Accuracy | **0.8765** |
| F1-score | **0.8798** |

Confusion Matrix:
[[849 151]
[ 96 904]]


Classification Report:
- Both classes are predicted correctly
- No class collapse
- Stable generalization

---

## 🚀 Inference Example

```bash
python src/predict.py --text "This movie was fantastic with great acting."
Output:


Prediction: positive (confidence=0.96)
Probabilities [negative, positive]: [0.04, 0.96]
🌐 API Deployment (FastAPI)
The model is deployed as a REST API using FastAPI.

Start the server

uvicorn app:app --host 127.0.0.1 --port 8000
Interactive API documentation
Open in browser:

http://127.0.0.1:8000/docs
Example API request (PowerShell)

Invoke-RestMethod -Method Post "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -Body '{"text":"Terrible movie. Waste of time."}'
Example response:

{
  "label": "negative",
  "confidence": 0.94,
  "probabilities": {
    "negative": 0.94,
    "positive": 0.06
  }
}
📁 Project Structure

bert-IMDb-sentiment/
│
├── src/
│   ├── dataset.py      # Custom PyTorch Dataset
│   ├── train.py        # Fine-tuning script
│   ├── eval.py         # Evaluation metrics
│   ├── predict.py      # CLI inference
│   └── infer.py        # Inference logic for API
│
├── app.py              # FastAPI application
├── requirements.txt
├── README.md
└── models/             # Generated during training (not tracked in Git)
🔁 Reproducibility
Model weights are not committed to the repository (best practice).

To reproduce results:

pip install -r requirements.txt
python src/train.py
python src/eval.py
🧩 Technologies Used
Python

PyTorch

Hugging Face Transformers

Hugging Face Datasets

FastAPI

Uvicorn

Scikit-learn

🔮 Future Improvements
Full dataset training

ONNX Runtime optimization for faster CPU inference

Multilingual sentiment analysis (e.g., Moroccan Darija)

Deployment on Hugging Face Spaces

Streamlit web interface

Authentication and rate limiting

👤 Author
Khalid Oqaidi
PhD in Computer Science & Artificial Intelligence
Research interests: Machine Learning, NLP, Responsible AI