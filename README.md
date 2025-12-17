# IMDb Sentiment Analysis with DistilBERT (Fine-Tuning)

This project demonstrates **fine-tuning a pretrained Transformer model (DistilBERT)** for binary sentiment classification (positive vs negative) on the **IMDb movie reviews dataset**.

The goal is to build an **end-to-end modern NLP pipeline** including:
- dataset handling
- fine-tuning a Transformer
- robust evaluation (Accuracy, F1, confusion matrix)
- inference on custom text

This project is designed as a **portfolio-ready example** of transfer learning and fine-tuning in NLP.

---

## 📌 Model & Dataset

- **Model**: `distilbert-base-uncased`
- **Dataset**: IMDb Movie Reviews  
  Source: Hugging Face `datasets` library
- **Task**: Binary sentiment classification (0 = negative, 1 = positive)

---

## 🧠 Methodology

1. Load IMDb dataset using Hugging Face `datasets`
2. Shuffle and select a **balanced subset** for fast iteration
3. Tokenize text using DistilBERT tokenizer
4. Fine-tune the pretrained Transformer
5. Evaluate using:
   - Accuracy
   - F1-score
   - Confusion Matrix
6. Run inference on unseen text

Dynamic padding is used via `DataCollatorWithPadding` for efficiency.

---

## 📊 Results (Small Subset)

Evaluation on a shuffled test subset (2,000 samples):

| Metric | Score |
|------|------|
| Accuracy | **0.8765** |
| F1-score | **0.8798** |

Confusion Matrix:
[[849 151]
[ 96 904]]


---

## 🚀 How to Run (Colab – GPU Recommended)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/oqaidik/bert-IMDb-sentiment.git
cd bert-IMDb-sentiment
2️⃣ Install dependencies

pip install -r requirements.txt
3️⃣ Train the model

python src/train.py
This will:

fine-tune DistilBERT

save the model to models/distilbert_imdb_small/

4️⃣ Evaluate the model

python src/eval.py
Outputs:

Accuracy

F1-score

Confusion matrix

Classification report

5️⃣ Run inference on custom text

python src/predict.py --text "This movie was surprisingly good with great acting."
Example output:


Prediction: positive (confidence=0.94)
Probabilities [neg, pos]: [0.06, 0.94]
📁 Project Structure

bert-IMDb-sentiment/
│
├── src/
│   ├── dataset.py      # Custom PyTorch Dataset
│   ├── train.py        # Fine-tuning script
│   ├── eval.py         # Evaluation metrics
│   └── predict.py      # Inference script
│
├── requirements.txt
├── README.md
└── models/             # Generated during training (not tracked in Git)
⚠️ Notes
Model checkpoints are not stored in GitHub (best practice).

Training can be re-run anytime to regenerate the model.

Designed for reproducibility and clarity.

🔮 Future Improvements
Full dataset training

Learning rate tuning

Model comparison (BERT, RoBERTa)

Multilingual sentiment analysis (e.g., Moroccan Darija)

Deployment with FastAPI or Streamlit

👤 Author
Khalid Oqaidi
PhD in Computer Science & Artificial Intelligence
Focus: Machine Learning, NLP, and Responsible AI