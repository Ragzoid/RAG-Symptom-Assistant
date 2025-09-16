# run_in_colab.py
"""
Colab-ready launcher for RAG Symptom Assistant.
Automatically uses backend/app.py chatbot_response() function.
Launches Gradio with share=True for a public URL.
"""

import sys
import os
import gradio as gr

# Add backend folder to Python path
BACKEND_DIR = os.path.join(os.getcwd(), "backend")
sys.path.insert(0, BACKEND_DIR)

# Import the chatbot_response function from app.py
try:
    from app import chatbot_response
except ImportError:
    raise ImportError(
        "Cannot import chatbot_response from backend/app.py. "
        "Ensure you applied the modification to define chatbot_response(user_input: str)."
    )

# Define Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(
        "# RAG Symptom Assistant (Demo)\n"
        "**Educational only — not medical advice.**"
    )

    chatbot = gr.Chatbot()
    txt = gr.Textbox(
        placeholder="Describe your symptom (e.g., 'I have fever and body ache')",
        lines=2
    )
    btn = gr.Button("Send")

    # function to handle user input
    def handle_input(message, chat_history):
        chat_history = chat_history or []
        chat_history.append(("User", message))
        answer = chatbot_response(message)
        chat_history.append(("Assistant", answer))
        return chat_history

    # link textbox & button to function
    btn.click(handle_input, inputs=[txt, chatbot], outputs=[chatbot])
    txt.submit(handle_input, inputs=[txt, chatbot], outputs=[chatbot])

# Launch Gradio with share=True for Colab public URL
demo.launch(share=True)
