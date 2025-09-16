# backend/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import os
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI(title="RAG Symptom Assistant (demo)")

KB_PATH = os.path.join("data", "knowledge_base_20.json")
COLLECTION_NAME = "conditions"

# load KB
with open(KB_PATH, "r") as f:
    KB = json.load(f)

# load embeddings & chroma
embed_model = SentenceTransformer("all-mpnet-base-v2")
client = chromadb.Client(Settings(anonymized_telemetry=False))
# create/load collection (rebuild if missing)
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    # build collection
    from backend.index import build_index
    collection = build_index(KB)

# generator - flan-t5-small (CPU fallback if no GPU)
GEN = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(GEN)
model = AutoModelForSeq2SeqLM.from_pretrained(GEN)
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    model = model.to("cuda")
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if device==0 else -1)

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

@app.post("/ask")
def ask(req: AskRequest):
    q = req.question
    q_emb = embed_model.encode([q], convert_to_numpy=True)[0].tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=req.top_k)
    candidates = []
    for i in range(len(results["ids"][0])):
        candidates.append({
            "id": results["ids"][0][i],
            "condition": results["metadatas"][0][i]["condition"],
            "doc": results["documents"][0][i],
            "score": results.get("distances", [[]])[0][i] if "distances" in results else None
        })
    # prepare a concise generator prompt referencing top candidate
    top_cond = candidates[0]["condition"]
    kb_entry = KB.get(top_cond, {})
    prompt = f"User question: {q}\nTop condition candidate: {top_cond}\nSymptoms: {', '.join(kb_entry.get('symptoms', []))}\nProvide a short educational summary and suggest Ayurvedic and English options (include dosage & quantity). Also include disclaimer."
    out = generator(prompt, max_length=256, do_sample=False)
    return {"answer": out[0]["generated_text"], "candidates": candidates}
