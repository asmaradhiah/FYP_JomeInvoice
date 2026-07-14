"""
Lightweight smoke-check to verify the project can start correctly without
performing heavy network downloads. It checks:
- Required environment variables (logs warnings if absent)
- Presence and writability of the project-local cache directory `.hf_cache`
- Imports of key Python packages used by the project
- Instantiation of embedding objects (no heavy network calls expected at init)

Run from project root:
    python scripts/run_smoke_check.py
"""

import os
import sys

ROOT = os.path.abspath(os.getcwd())
CACHE_DIR = os.path.join(ROOT, ".hf_cache")

print("Running HybridRAG smoke check...")

# 1) Environment variables
required_env = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
missing = [k for k in required_env if not os.getenv(k)]
if missing:
    print("WARNING: Missing environment variables:", ", ".join(missing))
else:
    print("All required environment variables found.")

# 2) Cache dir
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    testfile = os.path.join(CACHE_DIR, ".__write_test")
    with open(testfile, "w") as f:
        f.write("ok")
    os.remove(testfile)
    print(f"Cache directory OK: {CACHE_DIR}")
except Exception as e:
    print(f"ERROR: Cannot write to cache dir {CACHE_DIR}: {e}")
    sys.exit(2)

# 3) Optional imports
packages = [
    ("streamlit", "streamlit"),
    ("supabase", "supabase"),
    ("llama_index", "llama_index"),
    ("scipy", "scipy"),
    ("sklearn", "sklearn"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
]
missing_pkgs = []
for name, mod in packages:
    try:
        __import__(mod)
        print(f"Import OK: {name}")
    except Exception as e:
        print(f"Import FAIL: {name} -> {e}")
        missing_pkgs.append(name)

# 4) Instantiate embedding classes (lightweight) to ensure constructor args valid
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    print("HuggingFaceEmbedding available.")
    try:
        emb = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", model_kwargs={"cache_dir": CACHE_DIR})
        rer = HuggingFaceEmbedding(model_name="BAAI/bge-reranker-v2-m3", model_kwargs={"cache_dir": CACHE_DIR})
        print("Embedding constructors instantiated (no downloads triggered at init).")
    except Exception as e:
        print("Embedding instantiation error:", e)
except Exception as e:
    print("HuggingFaceEmbedding import failed:", e)

print("\nSmoke check complete.")
if missing_pkgs:
    print("Some optional packages are missing — install them to run full app:", ", ".join(missing_pkgs))
    sys.exit(3)

print("All basic checks passed (or warnings reported).\nYou can now run the app with Streamlit, but be aware large models will download if not already cached.")
