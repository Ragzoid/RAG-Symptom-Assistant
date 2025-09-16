# run_in_colab.py
"""
Colab-ready RAG Symptom Assistant
- Multi-turn clarifying questions
- Final prescription with Ayurvedic + English medicines
- Formatted output with educational summary and disclaimer
"""

import sys
import os
import gradio as gr

# Add backend folder to Python path
BACKEND_DIR = os.path.join(os.getcwd(), "backend")
sys.path.insert(0, BACKEND_DIR)

# Import from app.py
try:
    from app import embed_model, collection, KB, generator
except ImportError:
    raise ImportError(
        "Please ensure backend/app.py is updated with chatbot_response logic."
    )

# -----------------------
# Generate final answer
# -----------------------
def generate_answer(top_cond: str) -> str:
    """
    Generate the final response including medicines and summary.
    """
    top_cond_lower = top_cond.lower()
    kb_entry = KB.get(top_cond_lower, {})

    if not kb_entry:
        return f"Sorry, I don't have detailed information for {top_cond}."

    # Medicine lists
    ayurvedic_meds = kb_entry.get("ayurvedic", [])
    english_meds = kb_entry.get("english", [])

    ayurvedic_text = "\n".join([f"- {m['medicine']}: {m['dosage']} (Qty: {m['quantity']})"
                                for m in ayurvedic_meds]) or "No Ayurvedic medicine available."
    english_text = "\n".join([f"- {m['medicine']}: {m['dosage']} (Qty: {m['quantity']})"
                              for m in english_meds]) or "No English medicine available."

    symptoms_text = ", ".join(kb_entry.get("symptoms", []))

    prompt = (
        f"Top condition candidate: {top_cond}\n"
        f"Symptoms: {symptoms_text}\n\n"
        f"Ayurvedic medicines:\n{ayurvedic_text}\n\n"
        f"English medicines:\n{english_text}\n\n"
        "Provide a short educational summary and include a disclaimer."
    )

    out = generator(prompt, max_length=256, do_sample=False)
    return out[0]["generated_text"]

# -----------------------
# Multi-turn chat handler
# -----------------------
def handle_chat(message, chat_history, questions_asked, top_cond):
    chat_history = chat_history or []
    questions_asked = questions_asked or []

    # Determine top condition if first message
    if not top_cond:
        q_emb = embed_model.encode([message], convert_to_numpy=True)[0].tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=3)
        top_cond = results["metadatas"][0][0]["condition"]

    kb_entry = KB.get(top_cond.lower(), {})
    clarifying_questions = kb_entry.get("questions", [])

    # Ask next clarifying question if any
    next_q = None
    for q in clarifying_questions:
        if q not in questions_asked:
            next_q = q
            break

    chat_history.append(("User", message))

    if next_q:
        questions_asked.append(next_q)
        chat_history.append(("Assistant", next_q))
        return chat_history, questions_asked, top_cond
    else:
        # All questions asked → final prescription
        answer = generate_answer(top_cond)
        chat_history.append(("Assistant", answer))
        return chat_history, questions_asked, top_cond

# -----------------------
# Gradio interface
# -----------------------
with gr.Blocks() as demo:
    gr.Markdown(
        "# RAG Symptom Assistant (Demo)\n"
        "**Educational only — not medical advice.**"
    )

    chatbot = gr.Chatbot()
    txt = gr.Textbox(
        placeholder="Describe your symptoms (e.g., 'I have fever and body ache')",
        lines=2
    )
    btn = gr.Button("Send")

    # State for multi-turn chat
    state_chat = gr.State([])
    state_questions = gr.State([])
    state_topcond = gr.State(None)

    btn.click(
        handle_chat,
        inputs=[txt, state_chat, state_questions, state_topcond],
        outputs=[chatbot, state_questions, state_topcond]
    )
    txt.submit(
        handle_chat,
        inputs=[txt, state_chat, state_questions, state_topcond],
        outputs=[chatbot, state_questions, state_topcond]
    )

# Launch Gradio with public URL
demo.launch(share=True)
