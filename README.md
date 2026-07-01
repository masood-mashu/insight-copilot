---
title: Insight Copilot
emoji: 📊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.47.1
app_file: app/streamlit_app.py
pinned: false
---

# Insight Copilot

Insight Copilot is a multi-agent exploratory data analysis assistant built for a hackathon demo. Upload a CSV and the system profiles the dataset, generates plain-English insights, checks long-term vector memory for similar prior datasets, and recommends next analytical steps.

## Stack

- FastAPI backend
- Streamlit frontend
- Gemini via the supported `google-genai` SDK
- Qdrant Cloud for vector memory
- sentence-transformers for local embeddings
- Async orchestration with a Lyzr-compatible wrapper point in the pipeline

## Run locally

1. Create a `.env` file from `.env.example`
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. For the simplest single-process local or deployed demo, set:

```bash
USE_LOCAL_PIPELINE=true
```

Then start Streamlit:

```bash
streamlit run app/streamlit_app.py
```

4. For API mode, set `USE_LOCAL_PIPELINE=false`, start the API, then start Streamlit:

```bash
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

## Deployment

For Hugging Face Spaces or Streamlit Cloud, deploy only the Streamlit app and set `USE_LOCAL_PIPELINE=true` in secrets. The FastAPI app remains available in the repo for local API testing and technical review, but the hosted demo does not need a second process.

## API endpoints

- `GET /health`
- `POST /analyze`
- `GET /memory/history`

## Notes

- The first live run of the memory agent may download the `all-MiniLM-L6-v2` embedding model.
- Set `GEMINI_MODEL` in `.env` if you want to override the default Flash model name.
- `agents/insight_agent.py` uses the current `google-genai` client with schema-constrained JSON responses to keep outputs aligned with the required contract.
- In this workspace runtime, the published `lyzr` package is not installable because its available builds require Python versions below 3.12. The pipeline is written so a Lyzr wrapper can be added on a compatible runtime without restructuring the app.
