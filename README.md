# RAG Symptom Assistant (Educational demo)

This repository contains an end-to-end demo: knowledge base + vector DB + NLP embeddings + DL generator + Gradio UI.

**WARNING:** Demo only. Not medical advice. Always consult licensed clinicians.

## Quick start (test in Colab)

1. Push this repo to GitHub.
2. Open Google Colab.
3. Create a new notebook and run the single cell below (replace `<GITHUB_REPO_URL>` with your repo):

```python
# one-cell: clone + install + run
!git clone <GITHUB_REPO_URL> repo
%cd repo
!pip install -r requirements.txt --quiet
# run the integrated Colab runner (this will launch Gradio and show a public URL)
!python run_in_colab.py
