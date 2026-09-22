# Hugging Face Document Q&A

A lightweight local UI for asking questions about documents while running the large language model remotely through Hugging Face Inference Providers.

## Architecture

Browser
  -> Streamlit app on your laptop
  -> PDF/DOCX/TXT/MD text extraction on CPU
  -> local TF-IDF retrieval on CPU
  -> only the most relevant chunks + question
  -> Hugging Face Inference Providers
  -> `openai/gpt-oss-120b`
  -> answer returned to the Streamlit UI

The 120B model is NOT downloaded to your laptop.

## 1. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Create a Hugging Face token

Create a fine-grained token in Hugging Face settings and enable permission to make calls to Inference Providers.

Do not place your token in source code and do not commit it to Git.

## 3. Set the token

### Windows PowerShell

```powershell
$env:HF_TOKEN="hf_your_token_here"
```

### Windows cmd

```cmd
set HF_TOKEN=hf_your_token_here
```

### macOS / Linux

```bash
export HF_TOKEN="hf_your_token_here"
```

You can also paste the token into the password field in the app sidebar.

## 4. Run

```bash
streamlit run app.py
```

Streamlit will open a local web page, normally at:

```text
http://localhost:8501
```

## Model

The default is:

```text
openai/gpt-oss-120b:fastest
```

The `:fastest` suffix lets Hugging Face route the model to an available fast provider.

You can change the model in the sidebar.

## What happens to your document?

The full document is parsed locally. The app performs retrieval locally and sends only the selected relevant chunks, the question, and a short conversation history to the hosted model.

This version intentionally uses TF-IDF retrieval so it does not need a local embedding model or GPU.

## Important limitations

- Hosted inference is not guaranteed to be unlimited or permanently free.
- Your Hugging Face account may need available free credits or billing depending on the provider/model and usage.
- Scanned/image-only PDFs need OCR; this starter app only extracts embedded PDF text.
- TF-IDF is lexical retrieval. For production-quality semantic RAG, replace it with embeddings + a vector database.
- Never expose your HF token in frontend JavaScript. Keep model calls on the Python backend.
