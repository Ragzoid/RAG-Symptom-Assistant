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

# ---------------------------
# Paths & constants
# ---------------------------
KB_PATH = os.path.join("data", "knowledge_base_20.json")
COLLECTION_NAME = "conditions"

# ---------------------------
# Load knowledge base
# ---------------------------
with open(KB_PATH, "r") as f:
    KB = json.load(f)

# ---------------------------
# Load embeddings & Chroma
# ---------------------------
embed_model = SentenceTransformer("all-mpnet-base-v2")
client = chromadb.Client(Settings(anonymized_telemetry=False))

# create/load collection
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    # build collection if missing
    from backend.index import build_index
    collection = build_index(KB)

# ---------------------------
# Load generator (DL)
# ---------------------------
GEN = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(GEN)
model = AutoModelForSeq2SeqLM.from_pretrained(GEN)
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    model = model.to("cuda")

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == 0 else -1
)

# ---------------------------
# Function for Gradio/Colab
# ---------------------------
def chatbot_response(user_input: str) -> str:
    """
    Function for direct calling from Gradio/Colab.
    Takes user_input string and returns assistant's answer.
    """
    q = user_input
    q_emb = embed_model.encode([q], convert_to_numpy=True)[0].tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=3)

    # pick top candidate
    top_cond = results["metadatas"][0][0]["condition"]
    kb_entry = KB.get(top_cond, {})

    prompt = (
        f"User question: {q}\n"
        f"Top condition candidate: {top_cond}\n"
        f"Symptoms: {', '.join(kb_entry.get('symptoms', []))}\n"
        f"Provide a short educational summary and suggest Ayurvedic and English options "
        f"(include dosage & quantity). Also include disclaimer."
    )

    out = generator(prompt, max_length=256, do_sample=False)
    return out[0]["generated_text"]

# ---------------------------
# FastAPI endpoint (optional)
# ---------------------------
class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

@app.post("/ask")
def ask(req: AskRequest):
    answer = chatbot_response(req.question)
    return {"answer": answer}
