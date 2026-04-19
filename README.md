# NLP Assignment 2

This repository contains the complete implementation of Assignment 2 for Neural NLP Pipeline.

## Structure

- `embeddings/`
  - `tfidf_matrix.npy`
  - `ppmi_matrix.npy`
  - `embeddings_w2v.npy`
  - `word2idx.json`

- `models/`
  - `bilstm_pos.pt`
  - `bilstm_ner.pt`
  - `transformer_cls.pt`

- `data/`
  - `pos_train.conll`
  - `pos_test.conll`
  - `ner_train.conll`
  - `ner_test.conll`

- `figures/`
  - training curves
  - t-SNE plot
  - transformer plots

## Run order

1. Stage 0: prepare raw.txt, cleaned.txt, Metadata.json
2. Stage 1: build vocabulary and term-document matrix
3. Part 1: TF-IDF, PPMI, Word2Vec, evaluation
4. Part 2: POS tagging, NER, BiLSTM, CRF, ablations
5. Part 3: Transformer encoder for topic classification
6. Export report and zip folder

## Main notebook

`i22-0452_Assignment2_AI-8A.ipynb`

## Notes

- Implemented fully in PyTorch
- No pretrained models
- No HuggingFace
- No built-in Transformer classes