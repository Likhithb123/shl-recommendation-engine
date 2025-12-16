import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load catalog
df = pd.read_csv("data.csv")

# Combine important text columns
texts = (
    df["name"] + " " +
    df["test_type"] + " " +
    df["description"]
)

# Create embeddings using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
embeddings = vectorizer.fit_transform(texts)

# Save embeddings and metadata
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

df.to_pickle("catalog.pkl")

print("Embeddings created for", embeddings.shape[0], "assessments")
