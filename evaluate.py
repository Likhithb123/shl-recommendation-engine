import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load assessment catalog (prototype subset)
catalog = pd.read_pickle("catalog.pkl")

# Load vectorizer and embeddings
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Load SHL labeled dataset
df = pd.read_csv("datasets/shl_labeled.csv")

# Group ground-truth URLs by query
ground_truth = df.groupby("Query")["Assessment_url"].apply(list).to_dict()

def recommend(query, k=10):
    """Return top-k recommended assessment URLs for a query"""
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, embeddings)[0]
    ranked = scores.argsort()[::-1][:k]
    return catalog.iloc[ranked]["url"].tolist()

recalls = []
catalog_urls = set(catalog["url"])

for query, true_urls in ground_truth.items():
    # Keep only URLs that exist in the prototype catalog
    true_urls = [u for u in true_urls if u in catalog_urls]

    # Skip queries with no overlap
    if not true_urls:
        continue

    predicted = recommend(query, k=10)
    hits = len(set(predicted) & set(true_urls))
    recall = hits / len(true_urls)
    recalls.append(recall)

# Final safe output
if len(recalls) == 0:
    print(
        "Mean Recall@10 = 0.0 "
        "(No overlapping ground-truth URLs between SHL dataset and prototype catalog)"
    )
else:
    print("Mean Recall@10 =", sum(recalls) / len(recalls))
