from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# FastAPI app
# -----------------------------------
app = FastAPI(title="SHL Assessment Recommendation API")

# -----------------------------------
# Health check (IMPORTANT for Render)
# -----------------------------------
@app.get("/")
def health():
    return {"status": "ok"}

# -----------------------------------
# Request schema
# -----------------------------------
class QueryRequest(BaseModel):
    query: str

# -----------------------------------
# Lazy-loaded global objects
# -----------------------------------
catalog = None
vectorizer = None
embeddings = None

def load_models():
    """
    Load heavy files only once, when needed.
    Prevents slow startup on Render.
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

# -----------------------------------
# Recommendation endpoint
# -----------------------------------
@app.post("/recommend")
def recommend(request: QueryRequest):
    load_models()

    # Vectorize query
    query_vec = vectorizer.transform([request.query])

    # Ensure embeddings are numpy array
    emb = np.array(embeddings)

    # Compute cosine similarity
    scores = cosine_similarity(query_vec, emb)[0]

    # Get top 5 matches
    top_indices = np.argsort(scores)[::-1][:5]

    results = []

    for idx in top_indices:
        item = catalog[idx]

        # Handle different catalog formats safely
        if isinstance(item, dict):
            name = item.get("name")
            url = item.get("url")
            test_type = item.get("test_type")
            description = item.get("description")

        elif hasattr(item, "__dict__"):
            name = getattr(item, "name", None)
            url = getattr(item, "url", None)
            test_type = getattr(item, "test_type", None)
            description = getattr(item, "description", None)

        else:
            # Fallback for tuple / pandas row
            try:
                name = item[0]
                url = item[1]
                test_type = item[2] if len(item) > 2 else None
                description = item[3] if len(item) > 3 else None
            except Exception:
                continue

        results.append({
            "name": name,
            "url": url,
            "test_type": test_type,
            "description": description
        })

    return {"recommended_assessments": results}
