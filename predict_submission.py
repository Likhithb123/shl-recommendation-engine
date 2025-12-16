import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load catalog
catalog = pd.read_pickle("catalog.pkl")

# Load vectorizer and embeddings
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Load SHL dataset (queries)
df = pd.read_csv("datasets/shl_labeled.csv")

queries = df["Query"].unique()

rows = []

for query in queries:
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, embeddings)[0]
    ranked = scores.argsort()[::-1][:10]  # top 10

    for idx in ranked:
        rows.append({
            "Query": query,
            "Assessment_url": catalog.iloc[idx]["url"]
        })

# Save submission file
out = pd.DataFrame(rows)
out.to_csv("submission.csv", index=False)

print("submission.csv created with", len(out), "rows")
