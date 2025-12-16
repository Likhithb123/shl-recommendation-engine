from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI(title="SHL Assessment Recommendation API")

@app.get("/")
def health():
    return {"status": "ok"}


class QueryRequest(BaseModel):
    query: str


catalog = None
vectorizer = None
embeddings = None

def load_models():
    """
    Load heavy files only once, when needed.
    This avoids slow startup on Render.
    """
    global catalog, vectorizer, embeddings

    if catalog is None:
        with open("catalog.pkl", "rb") as f:
            catalog = pickle.load(f)

    if vectorizer is None:
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

    if embeddings is None:
        with open("embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)

# -----------------------
# Recommendation endpoint
# -----------------------
@app.post("/recommend")
def recommend(request: QueryRequest):
    # Load models only when this endpoint is called
    load_models()

    # Vectorize query
    query_vec = vectorizer.transform([request.query])

    # Compute similarity
    scores = cosine_similarity(query_vec, embeddings)[0]

    # Get top 5 matches
    top_indices = np.argsort(scores)[::-1][:5]

    results = []
    for idx in top_indices:
        item = catalog[idx]
        results.append({
            "name": item.get("name"),
            "url": item.get("url"),
            "test_type": item.get("test_type"),
            "description": item.get("description")
        })

    return {
        "recommended_assessments": results
    }
