# frontend/ui.py
import gradio as gr
import requests
import os
import json

# If you want to call the backend API (deployed), set API_URL, otherwise frontend launches integrated pipeline in Colab
API_URL = os.environ.get("RAG_API_URL", "")

def frontend_call_local(query):
    # if API_URL configured, call it
    if API_URL:
        resp = requests.post(API_URL + "/ask", json={"question": query})
        data = resp.json()
        return data.get("answer", "No answer"), data.get("candidates", [])
    return "This frontend is designed to call the local Colab integrated script. Launch the Colab launcher (run_in_colab.py) instead."

def run_frontend_local():
    with gr.Blocks() as demo:
        gr.Markdown("# RAG Symptom Assistant (Demo)\n**Educational only — not medical advice.**")
        chatbot = gr.Chatbot()
        txt = gr.Textbox(placeholder="Describe your symptom (e.g., 'I have fever and body ache')", lines=2)
        btn = gr.Button("Send")

        def user_submit(msg, chat_history):
            chat_history = chat_history or []
            chat_history.append(("User", msg))
            ans, _ = frontend_call_local(msg)
            chat_history.append(("Assistant", ans))
            return chat_history

        btn.click(user_submit, inputs=[txt, chatbot], outputs=[chatbot])
        txt.submit(user_submit, inputs=[txt, chatbot], outputs=[chatbot])
    demo.launch()
