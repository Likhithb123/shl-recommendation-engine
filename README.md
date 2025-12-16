# SHL Assessment Recommendation Engine

## Overview

This project implements an AI-powered recommendation engine that suggests suitable SHL assessments based on natural language hiring queries. The system is designed as part of the SHL Research Intern assignment.

## Problem Statement

Given a recruiter’s job description or hiring requirement, recommend the most relevant SHL assessments that evaluate the required technical and behavioral skills.

## Approach

### 1. Assessment Catalog

A representative subset of SHL assessments was used as the knowledge base, containing:

- Assessment name
- URL
- Test type (Knowledge & Skills / Personality & Behavior)
- Description

### 2. Text Embeddings

- Combined assessment name, test type, and description
- Used TF-IDF vectorization
- Generated embeddings for semantic similarity search

### 3. Recommendation Logic

- User query is vectorized
- Cosine similarity is computed against assessment embeddings
- Top-K most similar assessments are retrieved
- Balanced recommendations ensure both technical and behavioral assessments

### 4. API

A FastAPI-based REST API exposes the recommendation functionality via a `/recommend` endpoint.

### 5. Evaluation

The provided labeled dataset was used to evaluate the system using Mean Recall@10.
Due to the use of a representative subset of the SHL catalog, no overlapping URLs existed between the labeled dataset and the prototype catalog, resulting in a Recall@10 of 0.0.
The evaluation pipeline is valid and would yield meaningful scores when applied to the full catalog.

## Files

- `app.py` – FastAPI application
- `build_embeddings.py` – Embedding generation
- `evaluate.py` – Evaluation script
- `predict_submission.py` – Submission file generator
- `data.csv` – Assessment catalog
- `datasets/shl_labeled.csv` – Provided labeled dataset
- `submission.csv` – Final predictions

## How to Run

```bash
python build_embeddings.py
python -m uvicorn app:app --reload
python evaluate.py
python predict_submission.py
```
