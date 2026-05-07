# Assignment 4 — Transformers, Attention, SHAP & LIME

Binary sentiment classification on the Amazon Polarity dataset using fine-tuned BERT, with attention visualisation and explainability via SHAP and LIME.

---

## Project Structure

```
assignment4/
├── assignment4_master.ipynb   # Single master notebook (all steps)
├── Assignment4_Report.pdf     # Full written report (12 pages)
├── README.md                  # This file
└── data/                      # Auto-created by notebook
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── test_with_preds.csv
```

---

## Requirements

- Google Account (for Google Colab)
- No local installation needed — everything runs in Colab

---

## How to Run

### Step 1 — Open in Google Colab

Upload `assignment4_master.ipynb` to [https://colab.research.google.com](https://colab.research.google.com)  
or open it directly from Google Drive.

### Step 2 — Enable GPU

```
Runtime > Change runtime type > Hardware accelerator > GPU (T4)
```

### Step 3 — Run All Cells

```
Runtime > Run all
```

That's it. The notebook handles everything from data download to explainability plots.

---

## What the Notebook Does

| Section | Description |
|---------|-------------|
| **1. Install & Imports** | Installs all packages, sets seed=42 |
| **2. Data Pipeline** | Loads Amazon Polarity, cleans text, tokenises, splits 70/15/15 |
| **3. BERT Fine-Tuning** | Fine-tunes bert-base-uncased for 3 epochs with early stopping |
| **4. Evaluation** | Accuracy, Precision, Recall, F1 + confusion matrix |
| **5. Attention Analysis** | Heatmaps from layers 1, 6, 11 across multiple heads |
| **6. SHAP** | Token-level explanations for 20 test samples |
| **7. LIME** | Word-level explanations for 20 test samples |
| **8. SHAP vs LIME** | Faithfulness, stability, and runtime comparison |
| **9. Error Analysis** | Misclassified sample inspection with LIME |

---

## Dataset

- **Source:** [fancyzhx/amazon_polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity)
- **Subset used:** 20,000 samples (10,000 Negative + 10,000 Positive)
- **Downloaded automatically** via HuggingFace `datasets` library — no manual download needed

---

## Model

| Setting | Value |
|---------|-------|
| Base model | `bert-base-uncased` |
| Epochs | 3 (early stopping, patience=2) |
| Learning rate | 2e-5 with linear warmup |
| Batch size | 32 |
| Max token length | 128 |
| Optimizer | AdamW (weight decay=0.01) |
| Expected accuracy | ~94% on test set |

---

## Hardware & Software

| Item | Value |
|------|-------|
| Platform | Google Colab (free tier) |
| GPU | NVIDIA Tesla T4 |
| Python | 3.10 |
| PyTorch | 2.x |
| Transformers | 4.x (HuggingFace) |
| SHAP | 0.44+ |
| LIME | 0.2.0.1 |

---

## Expected Runtime

| Stage | Time (T4 GPU) |
|-------|--------------|
| Data pipeline | ~3 min |
| BERT training (3 epochs) | ~25–30 min |
| Attention analysis | ~1 min |
| SHAP (20 samples) | ~5–8 min |
| LIME (20 samples) | ~3–5 min |
| **Total** | **~35–45 min** |

---

## Reproducibility

All random seeds are fixed at **42** for Python, NumPy, PyTorch, and CUDA.  
Running `Runtime > Run all` on a fresh Colab session will reproduce all results.

---

## References

1. Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers. NAACL-HLT.
2. Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions. NeurIPS.
3. Ribeiro et al. (2016) — Why Should I Trust You? KDD.
4. Vaswani et al. (2017) — Attention Is All You Need. NeurIPS.
5. Dataset: https://huggingface.co/datasets/fancyzhx/amazon_polarity
6. SHAP docs: https://shap.readthedocs.io/en/latest/text_examples.html
7. LIME docs: https://lime-ml.readthedocs.io/en/latest/
