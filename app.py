from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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
# Lazy-loaded globals
# --------------------
catalog = None
vectorizer = None
embeddings = None

def load_models():
    
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

# --------------------
# Recommendation endpoint (FULL FUNCTION)
# --------------------
@app.post("/recommend")
def recommend(request: QueryRequest):
    load_models()

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Vectorize query
    query_vec = vectorizer.transform([request.query])

    # Convert embeddings to numpy array
    emb = np.array(embeddings)

    # Compute similarity
    scores = cosine_similarity(query_vec, emb)[0]

    # Top 5 matches
    top_indices = np.argsort(scores)[::-1][:5]

    results = []

    for idx in top_indices:
        item = catalog[idx]

        
        if isinstance(item, dict):
            results.append({
                "name": item.get("name"),
                "url": item.get("url"),
                "test_type": item.get("test_type"),
                "description": item.get("description")
            })
        else:
            results.append({
                "name": str(item[0]) if len(item) > 0 else None,
                "url": str(item[1]) if len(item) > 1 else None,
                "test_type": str(item[2]) if len(item) > 2 else None,
                "description": str(item[3]) if len(item) > 3 else None
            })

    return {"recommended_assessments": results}
