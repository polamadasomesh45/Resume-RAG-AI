# ResumeRAG AI

## Run locally

1. `python3 -m venv .venv`
2. `source .venv/bin/activate` (Windows: `.venv\\Scripts\\activate`)
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
5. `streamlit run app.py`

## Notes
- Python 3.10+ recommended.
- This is a single-user demo, not a secure multi-tenant production service.
- PDF text must be selectable; scanned/image-only PDFs require OCR.
- Resume text is sent to Google's Gemini API for embeddings and responses.
- The vector database is in-memory per Streamlit session; restarting/resetting clears it.
- Source excerpts are shown for verification; citations from the model can still be inaccurate.
- API usage may incur charges or quotas.
