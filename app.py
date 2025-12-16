from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------
# App setup
# --------------------
app = FastAPI(title="SHL Assessment Recommendation API")

# --------------------
# Health check
# --------------------
@app.get("/")
def health():
    return {"status": "ok"}

# --------------------
# Request schema
# --------------------
class QueryRequest(BaseModel):
    query: str

# --------------------
# Global objects (built once)
# --------------------
df = None
vectorizer = None
embeddings = None

def load_data():
    """
    Load CSV and build vectorizer + embeddings once.
    This avoids pickle incompatibility issues.
    """
    global df, vectorizer, embeddings

    if df is None:
        df = pd.read_csv("data.csv")

        # Safety check
        required_cols = {"name", "url", "test_type", "description"}
        if not required_cols.issubset(df.columns):
            raise RuntimeError("data.csv missing required columns")

        # Combine text fields for similarity
        df["combined_text"] = (
            df["name"].fillna("") + " " +
            df["test_type"].fillna("") + " " +
            df["description"].fillna("")
        )

        vectorizer = TfidfVectorizer(stop_words="english")
        embeddings = vectorizer.fit_transform(df["combined_text"])

# --------------------
# Recommendation endpoint
# --------------------
@app.post("/recommend")
def recommend(request: QueryRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    load_data()

    # Vectorize query
    query_vec = vectorizer.transform([request.query])

    # Compute similarity
    scores = cosine_similarity(query_vec, embeddings)[0]

    # Top 5 matches
    top_indices = np.argsort(scores)[::-1][:5]

    results = []

    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            "name": row["name"],
            "url": row["url"],
            "test_type": row["test_type"],
            "description": row["description"]
        })

    return {"recommended_assessments": results}
