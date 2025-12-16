from fastapi import FastAPI
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load data
df = pd.read_pickle("catalog.pkl")

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/recommend")
def recommend(payload: dict):
    query = payload["query"]

    # Vectorize query
    query_vec = vectorizer.transform([query])

    # Compute similarity
    scores = cosine_similarity(query_vec, embeddings)[0]

    # Get top 5 results
    top_indices = scores.argsort()[-5:][::-1]

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
