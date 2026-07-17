# HybridRAG

HybridRAG is a Streamlit-based assistant for SME e-invoicing guidance. It uses a retrieval-augmented workflow with Supabase, Groq, and Hugging Face embeddings to answer questions based on a stored knowledge base.

## Features
- Chat-based assistant interface
- Semantic retrieval from Supabase documents
- LLM-generated answers using Groq
- Streamlit web app deployment support

## Project structure
- app.py — Streamlit entry point
- views/chatbot.py — chatbot UI and inference logic
- db_setup.py — database setup helpers
- evaluate.py — evaluation utilities
- requirements.txt — Python dependencies
- data/ and dataset/ — knowledge base and evaluation data

## Setup
1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with the required environment variables:
   ```env
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   GROQ_API_KEY=your_groq_api_key
   JINA_API_KEY=your_jina_api_key
   ```
4. Run the app locally:
   ```bash
   streamlit run app.py
   ```

Credit:
https://bge-model.com/

## Deployment
For Streamlit Community Cloud, deploy this repository and add the same environment variables in Streamlit Cloud Secrets.

## Notes
- Keep API keys and secrets out of version control.
- The repository includes a `.gitignore` file to avoid committing local environment files and cache directories.
