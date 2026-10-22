# Riiid! Answer Correctness Prediction

An encoder-decoder Transformer in PyTorch that predicts whether a student will answer their next question correctly, trained on 100M+ interactions from the [Riiid! Kaggle competition](https://www.kaggle.com/c/riiid-test-answer-prediction).

![Python](https://img.shields.io/badge/Python-3-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-TPU%20via%20torch__xla-EE4C2C) ![Kaggle](https://img.shields.io/badge/Kaggle-top%203%25-20BEFF)

## Results

- Competition finish: **top 3%** of teams (silver medal).
- Validation ROC AUC of **0.79** in this notebook, logged per epoch in the training cell.

## Problem

This is a knowledge-tracing task: model what a student knows over time from their history on an education app (TOEIC preparation). Each row is one interaction: a question answered or a lecture watched, with timing and whether the student saw the previous explanation. The target is `answered_correctly` for the next question. It was a time-series code competition, so predictions had to be served through Kaggle's streaming API in order, without looking ahead.

## Approach

- **Sequence model.** Each student's history becomes a window of up to 257 interactions. The encoder sums embeddings of the question id, TOEIC part, two question tags, prior elapsed time, whether an explanation was shown, time since the previous interaction, and the shifted previous answer. The decoder embeds the question and the previous answer and attends to the encoder output. Both use a causal mask, so each prediction only sees the past.
- **Architecture.** 2 encoder and 2 decoder layers, `d_model` 256, 8 heads, feed-forward 2048, GELU, dropout 0.1, learned position embeddings, sigmoid output per step.
- **Feature engineering.** Questions are joined with their metadata, raw timestamps become per-user time deltas, and continuous times are bucketed and capped before embedding.
- **Training.** Learning rate 1e-4, `ReduceLROnPlateau` on validation AUC, early stopping after 10 epochs without improvement, batch size 64, on a Colab TPU through `torch_xla`. Data is cached as Parquet for fast reloads.

## Repository layout

```
riiid_transformer.ipynb   data prep, model, training loop, and inference
requirements.txt
```

## Running it

1. Download the competition data from [Kaggle](https://www.kaggle.com/c/riiid-test-answer-prediction/data) (Kaggle account and rules acceptance required).
2. Open `riiid_transformer.ipynb` in Google Colab, or locally after `pip install -r requirements.txt`.
3. Point `DRIVE_DIR` in the *Global Variables* cell at the folder that holds the data. On a GPU or CPU runtime the notebook falls back from TPU automatically.

## Tech stack

Python, PyTorch, torch_xla (TPU), pandas, NumPy, scikit-learn, Google Colab.

## Author

[Moeen Bagheri](https://moeenbagheri.com)
