import requests  # HTTP calls for Jina reranker
import streamlit as st  # streamlit UI
import os  # environment and path utilities
import json  # json parsing
import html  # escape HTML in chat bubbles
import re  # lightweight markdown formatting
import base64  # embed local avatar image in HTML bubbles
from dotenv import load_dotenv  # load environment variables from .env
from supabase import create_client, Client  # supabase client types
from llama_index.llms.groq import Groq  # Groq LLM wrapper
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # embedding model
from scipy.spatial.distance import cosine  # cosine similarity function


load_dotenv()  # load environment variables

# Project-local HuggingFace cache directory to avoid global side-effects and deprecation warnings
CACHE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".hf_cache"))
os.makedirs(CACHE_DIR, exist_ok=True)

#st.title("")  # page title
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("assets/header.png", width=500)
#st.write("SME E-Invoicing Guideline Assistant")  # description

# --- 1. SETUP SUPABASE & MODEL ---
supabase_url = os.getenv("SUPABASE_URL")  # get Supabase URL
supabase_key = os.getenv("SUPABASE_KEY")  # get Supabase key
supabase: Client = create_client(supabase_url, supabase_key)  # initialize client

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"cache_dir": CACHE_DIR}
)  # initialize embedding model with project cache
llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))  # initialize LLM


def get_avatar_data_uri():
    avatar_path = os.path.join(os.path.dirname(__file__), "..", "assets", "mike.jpg")
    try:
        with open(avatar_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except FileNotFoundError:
        return "https://via.placeholder.com/32/1E90FF/FFFFFF?text=M"


def render_chat_message(role, content, metadata=None):
    if role == "user":
        escaped = html.escape(content).replace("\n", "<br>")
        bubble = f"""
        <div style=\"display:flex; justify-content:flex-end; margin:8px 0;\">
          <div style=\"background:#004d88; color:#fff; padding:12px 16px; border-radius:18px 18px 4px 18px; max-width:70%; white-space:pre-wrap; line-height:1.5;\">{escaped}</div>
        </div>
        """
    else:
        escaped = html.escape(content)
        rendered = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", escaped)
        rendered = rendered.replace("\n", "<br>")
        avatar_uri = get_avatar_data_uri()
        bubble = f"""
        <div style=\"display:flex; justify-content:flex-start; margin:8px 0;\">
          <div style=\"background:#f1f0f0; color:#111; padding:12px 16px; border-radius:18px 18px 18px 4px; max-width:70%; line-height:1.5;\">
            <div style=\"display:flex; align-items:flex-start; gap:12px;\">
              <img src=\"{avatar_uri}\" alt=\"Assistant\" style=\"width:32px;height:32px;border-radius:50%;flex-shrink:0;\" />
              <div style=\"min-width:0;\">{rendered}</div>
            </div>
          </div>
        </div>
        """
    st.markdown(bubble, unsafe_allow_html=True)
    if metadata and role == "assistant":
        st.caption(metadata)


def rerank_context_with_jina(query, retrieved_chapters):
    """Send the top retrieved chunks to Jina and return the top 3 reranked chunks with metadata."""
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('JINA_API_KEY')}",
    }

    documents_to_rerank = [chunk[1].get("text", "") for chunk in retrieved_chapters]
    payload = {
        "model": "jina-reranker-v3",
        "query": query,
        "documents": documents_to_rerank,
        "top_n": 3,
        "max_chunks_per_doc": 2048,
        "return_documents": True,
        "truncation": True,
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json()

        reranked_items = []
        for item in results.get("results", []):
            original_index = item["index"]
            sim, row, _ = retrieved_chapters[original_index]
            reranked_items.append({
                "text": documents_to_rerank[original_index],
                "source": row.get("source"),
                "page": row.get("page"),
                "similarity": sim,
            })
        return reranked_items

    except Exception as e:
        st.warning(f"⚠️ Jina Rerank failed: {e}. Falling back to raw vector sort.")
        fallback = []
        for sim, row, _ in retrieved_chapters[:3]:
            fallback.append({
                "text": row.get("text", ""),
                "source": row.get("source"),
                "page": row.get("page"),
                "similarity": sim,
            })
        return fallback


# --- 2. INSIALISASI MEMORI CHAT ---
if "messages" not in st.session_state:  # ensure chat history exists
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "**Hi, I’m Mika. You can ask me about:**\n\n1. E‑invoicing requirements for SMEs\n 2. Mandatory fields in e‑invoices\n 3. How to handle rejected e‑invoices\n 4. Compliance checks and procedures",
        }
    ]

# Papar sejarah chat lama
for message in st.session_state.messages:  # display previous messages
    render_chat_message(message["role"], message["content"], message.get("metadata"))

# --- 3. PROSES INPUT USER ---
user_input = st.chat_input("Ask a question about LHDN e-invoicing guideline here...")  # chat input box

if user_input:
    # Papar soalan user
    render_chat_message("user", user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})  # save to history
    
    # AI mula proses
    with st.spinner("💡 Finding the best answer..."):
            try:
                # A. Tukar soalan user jadi embedding vektor
                query_embedding = embed_model.get_text_embedding(user_input)  # compute query embedding
                
                # B. Tarik data dari Supabase
                response_db = supabase.table("documents").select("*, embedding").execute()  # fetch documents
                
                candidates = []  # list of (sim, row, db_emb)
                for row in response_db.data:  # iterate DB rows
                    db_emb = row.get("embedding")  # get stored embedding
                    if isinstance(db_emb, str):
                        db_emb = json.loads(db_emb)
                    if not db_emb:
                        continue
                    sim = 1 - cosine(query_embedding, db_emb)
                    candidates.append((sim, row, db_emb))

                if not candidates:
                    error_msg = "❌ I couldn't find any entries in the cloud database to score."
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[:25]
                    reranked_items = rerank_context_with_jina(user_input, candidates)

                    context_string = "\n\n---\n\n".join([item["text"] for item in reranked_items])
                    prompt_sources = " | ".join([
                        f"Source {chr(65 + idx)}: {item['source']} (p.{item['page']})"
                        for idx, item in enumerate(reranked_items)
                        if item.get("source") and item.get("page") is not None
                    ])
                    top_confidence = reranked_items[0]["similarity"] if reranked_items else 0.0
                    system_prompt = f"""
                        You are JomeInvoice AI, an expert consultant for Malaysia's LHDN e-Invoicing system.
                        Formulate a clean, comprehensive response answering the user's actual query.
                        You MUST base your response on the verified dataset reference and full background context text provided below.

                        VERIFIED CLOUD REFERENCE:
                        {context_string}

                        USER ACTUAL QUERY: {user_input}
                        """
                    response_ai = llm.complete(system_prompt)
                    ai_response = response_ai.text
                    metadata_text = (
                        f"📚 {prompt_sources} | Confidence: {top_confidence:.2%}"
                        if prompt_sources
                        else f"Confidence: {top_confidence:.2%}"
                    )

                    render_chat_message("assistant", ai_response, metadata_text)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ai_response,
                        "metadata": metadata_text
                    })
                    
            except Exception as e:
                st.error(f"Error querying cloud database: {e}")  # show exception