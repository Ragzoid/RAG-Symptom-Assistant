# backend/index.py
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os

KB_PATH = os.path.join("data", "knowledge_base_20.json")
COLLECTION_NAME = "conditions"

def load_kb(path=KB_PATH):
    with open(path, "r") as f:
        kb = json.load(f)
    return kb

def build_index(kb):
    print("Loading embedding model...")
    embed_model = SentenceTransformer("all-mpnet-base-v2")
    texts, metadatas, ids = [], [], []
    for idx, (cond, info) in enumerate(kb.items()):
        text = cond + ". Symptoms: " + "; ".join(info.get("symptoms", []))
        text += ". Questions: " + " | ".join(info.get("questions", []))
        meds = []
        for t in info.get("ayurvedic", []) + info.get("english", []):
            meds.append(f"{t.get('medicine')} ({t.get('dosage')})")
        if meds:
            text += ". Treatments: " + "; ".join(meds)
        texts.append(text)
        metadatas.append({"condition": cond})
        ids.append(f"cond_{idx}")

    print("Encoding embeddings...")
    embs = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    client = chromadb.Client(Settings(anonymized_telemetry=False))
    # delete if exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embs.tolist())
    print(f"Indexed {len(ids)} conditions into Chroma collection '{COLLECTION_NAME}'")
    return collection

if __name__ == "__main__":
    kb = load_kb()
    build_index(kb)
