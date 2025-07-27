import numpy as np
import faiss
from create_embeddings import encode_texts

save_path = "clip_text_embeddings.npz"

data = np.load(save_path)
files = data["files"]
folders = data["folders"]
embeddings = data["embeddings"]

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # Inner product (equivalent to cosine similarity on normalized vectors)
index.add(embeddings)

# ====== 5. Search Function ======
def search(query: str, top_k: int = 1):
    query_emb = encode_texts([query])
    scores, indices = index.search(query_emb, top_k)
    return folders[indices], files[indices]

# ====== 6. Run Search ======
if __name__ == "__main__":
    while True:
        query = input("\nEnter your query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        folders_, files_ = search(query)
        print(folders_, files_)