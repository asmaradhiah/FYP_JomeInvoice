import requests  # HTTP calls for Jina reranker
import streamlit as st  # streamlit UI
import os  # environment and path utilities
import json  # json parsing
import html  # escape HTML in chat bubbles
import re  # lightweight markdown formatting
import base64  # embed local avatar image in HTML bubbles
from dotenv import load_dotenv  # load environment variables from .env
from supabase import create_client, Client  # supabase client types
from scipy.spatial.distance import cosine  # cosine similarity function

try:
    from llama_index.llms.groq import Groq  # Groq LLM wrapper
except ModuleNotFoundError:  # pragma: no cover - handled for Streamlit Cloud compatibility
    Groq = None

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # embedding model
except ModuleNotFoundError:  # pragma: no cover - handled for Streamlit Cloud compatibility
    HuggingFaceEmbedding = None


load_dotenv()  # load environment variables


def get_secret(key, default=None):
    value = os.getenv(key)
    if value:
        return value
    try:
        return st.secrets[key]
    except Exception:
        return default

# Project-local HuggingFace cache directory to avoid global side-effects and deprecation warnings
CACHE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".hf_cache"))
os.makedirs(CACHE_DIR, exist_ok=True)


def get_banner_data_uri():
    banner_path = os.path.join(os.path.dirname(__file__), "..", "assets", "header.png")
    try:
        with open(banner_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        return "https://via.placeholder.com/770x220"

# prepare banner data URI for embedded HTML
banner_data_uri = get_banner_data_uri()

#st.title("")  # page title
# Inject Google Forms-like CSS and render a survey-style banner + content card
st.markdown("""
    <style>
    /* Hide default Streamlit header/footer elements for a cleaner form look */
    /*#MainMenu, header, footer {visibility: hidden;}
    
    /* Set the warm, off-white background of the page */
    .stApp {
        background-color: #fbf8f3;
    }
    
    /* Container wrapper */
    .form-container {
        max-width: 770px;
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        gap: 12px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Style the Streamlit main block as the chat card area */
    [data-testid="stMainBlockContainer"] {
        background-color: #ffffff;
        border: 1px solid #dadce0;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.1);
        padding: 18px;
        max-width: 770px;
        margin: 12px auto;
        display: flex;
        flex-direction: column;
    }

    /* The Streamlit main block will act as the chat area (no separate chat-body wrapper) */
    /* Messages are rendered by Streamlit and styled via the render_chat_message HTML bubbles. */

    /* Common Card Styling */
    .card {
        background-color: #ffffff;
        border: 1px solid #dadce0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.1);
    }

    /* Banner Card styling */
    .banner-card {
        border-top: 10px solid #203a43; /* Dark accent color */
        height: 220px; /* increased banner height */
    }
    .banner-card img {
        width: 100%;
        height: 220px; /* match banner height */
        display: block;
        object-fit: cover;
    }

    /* Content Card styling */
    .content-card {
        padding: 24px;
        color: #202124;
    }
    
    .form-title {
        font-size: 32px;
        font-weight: 400;
        margin-bottom: 8px;
        color: #202124;
    }
    
    .subtitle {
        font-size: 15px;
        margin-bottom: 16px;
        color: #202124;
    }
    
    .divider {
        border: 0;
        height: 1px;
        background-color: #dadce0;
        margin: 20px 0;
    }
    
    .form-body p {
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Render the Google Forms-like banner + content container
st.markdown(f"""
<div class="form-container">
    <!-- 1. Header Banner Card -->
    <div class="card banner-card">
        <!-- Use local banner if available -->
        <img src="{banner_data_uri}" alt="JomeInvoice Banner">
    </div>

</div>
""", unsafe_allow_html=True)

# --- 1. SETUP SUPABASE & MODEL ---
supabase_url = get_secret("SUPABASE_URL")  # get Supabase URL
supabase_key = get_secret("SUPABASE_KEY")  # get Supabase key
supabase: Client | None = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None  # initialize client

embed_model = None
llm = None

if Groq is not None:
    llm = Groq(model="llama-3.1-8b-instant", api_key=get_secret("GROQ_API_KEY"))  # initialize LLM


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
                    <div style=\"background:#e8f0fe; color:#202124; padding:12px 16px; border-radius:12px 12px 4px 12px; max-width:75%; white-space:pre-wrap; line-height:1.5; border:1px solid #dfe7ff;\">{escaped}</div>
                </div>
                """
        else:
                escaped = html.escape(content)
                rendered = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", escaped)
                rendered = rendered.replace("\n", "<br>")
                avatar_uri = get_avatar_data_uri()
                bubble = f"""
                <div style=\"display:flex; justify-content:flex-start; margin:8px 0;\">
                    <div style=\"background:#ffffff; color:#111; padding:12px 16px; border-radius:12px 12px 12px 4px; max-width:75%; line-height:1.5; border:1px solid #e6e6e6;\">
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
            "content": "**Hi, I’m Mika. You can ask me about:**\n\n- E‑invoicing requirements for SMEs\n- Mandatory fields in e‑invoices\n- How to handle rejected e‑invoices\n- Compliance checks and procedures\n and any questions related to e-invoice guideline.",
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

    if not supabase or not llm:
        error_msg = "⚠️ The app is missing required configuration. Please verify the Streamlit Cloud secrets for SUPABASE_URL, SUPABASE_KEY, and GROQ_API_KEY."
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        if embed_model is None and HuggingFaceEmbedding is not None:
            try:
                embed_model = HuggingFaceEmbedding(
                    model_name="BAAI/bge-small-en-v1.5",
                    model_kwargs={"cache_dir": CACHE_DIR}
                )
            except Exception as exc:
                st.error(f"Failed to initialize embedding model: {exc}")
                st.session_state.messages.append({"role": "assistant", "content": f"Failed to initialize embedding model: {exc}"})
                embed_model = None

        if embed_model is None:
            error_msg = "⚠️ The embedding model could not be initialized. Please retry or check the deployment environment."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
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