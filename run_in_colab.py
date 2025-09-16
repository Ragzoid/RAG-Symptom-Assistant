# run_in_colab.py
"""
Colab-ready RAG Symptom Assistant
- Multi-turn clarifying questions (from KB)
- Final prescription (Ayurvedic + English)
- Formatted output
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

# Helper function to generate final answer
def generate_answer(user_input, top_cond):
    kb_entry = KB.get(top_cond, {})

    ayurvedic_meds = kb_entry.get("ayurvedic", [])
    english_meds = kb_entry.get("english", [])

    ayurvedic_text = "\n".join([f"- {m['medicine']}: {m['dosage']} (Qty: {m['quantity']})" for m in ayurvedic_meds])
    english_text = "\n".join([f"- {m['medicine']}: {m['dosage']} (Qty: {m['quantity']})" for m in english_meds])

    symptoms_text = ", ".join(kb_entry.get("symptoms", []))

    prompt = (
        f"User question: {user_input}\n"
        f"Top condition candidate: {top_cond}\n"
        f"Symptoms: {symptoms_text}\n\n"
        f"Provide a short educational summary and format the following medications:\n\n"
        f"Ayurvedic medicines:\n{ayurvedic_text}\n\n"
        f"English medicines:\n{english_text}\n\n"
        "Also include a short disclaimer."
    )

    out = generator(prompt, max_length=256, do_sample=False)
    return out[0]["generated_text"]

# Multi-turn chat handler
def handle_chat(message, chat_history, questions_asked, top_cond):
    chat_history = chat_history or []
    questions_asked = questions_asked or []

    # Add user message
    chat_history.append(("User", message))

    # Get top candidate from RAG if first message
    if not top_cond:
        q_emb = embed_model.encode([message], convert_to_numpy=True)[0].tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=3)
        top_cond = results["metadatas"][0][0]["condition"]

    kb_entry = KB.get(top_cond, {})
    clarifying_questions = kb_entry.get("questions", [])

    # Ask next clarifying question if not done
    next_q = None
    for q in clarifying_questions:
        if q not in questions_asked:
            next_q = q
            break

    if next_q:
        questions_asked.append(next_q)
        chat_history.append(("Assistant", next_q))
        return chat_history, questions_asked, top_cond
    else:
        # All questions asked, provide final prescription
        answer = generate_answer(message, top_cond)
        chat_history.append(("Assistant", answer))
        return chat_history, questions_asked, top_cond

# Gradio interface
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

    # State for multi-turn
    state_chat = gr.State([])
    state_questions = gr.State([])
    state_topcond = gr.State(None)

    # Hook inputs to function
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
