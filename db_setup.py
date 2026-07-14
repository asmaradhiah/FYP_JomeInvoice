import os  # os utilities
import json  # json read/write
from dotenv import load_dotenv  # load .env variables
from supabase import create_client, Client  # supabase client
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # embedding model

load_dotenv()  # load environment variables from .env

supabase_url = os.getenv("SUPABASE_URL")  # get Supabase URL
supabase_key = os.getenv("SUPABASE_KEY")  # get Supabase key
supabase: Client = create_client(supabase_url, supabase_key)  # initialize Supabase client

# Project-local HuggingFace cache directory to avoid global cache side effects
CACHE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".hf_cache"))
os.makedirs(CACHE_DIR, exist_ok=True)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"cache_dir": CACHE_DIR}
)  # init embedding model with project cache

def upload_dataset_to_supabase(json_path):
    if not os.path.exists(json_path):  # ensure file exists
        print(f"❌ File not found: {json_path}")
        return

    print(f"📖 Reading data from {json_path}...")  # log read
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # parse JSON
        if isinstance(data, dict):
            data = [data]  # normalize single-object files to list

    print(f"🚀 Processing {len(data)} entries for Supabase...")  # log count
    
    batch = []  # collect rows for bulk insert
    batch_size = 100  # Upload 100 rows at once instead of 1 by 1
    
    for idx, item in enumerate(data):
        question = item.get("question", "")  # get question text
        if not question:
            continue  # skip entries without question
            
        print(f"[{idx+1}/{len(data)}] Embedding: {question[:40]}...")  # progress log
        try:
            embedding = embed_model.get_text_embedding(question)  # create embedding
        except Exception as e:
            print(f"⚠️ Skipping row {idx} due to embedding failure: {e}")
            continue  # skip problematic rows
        
        row = {
            "pair_id": item.get("pair_id"),  # optional pair id
            "source": item.get("source"),  # source doc
            "source_uri": item.get("source_uri"),  # source URI
            "page": item.get("page"),  # page number
            "chunk_index": item.get("chunk_index"),  # chunk idx
            "question": question,  # question text
            "answer": item.get("answer"),  # ground truth answer
            "text": item.get("text"),  # context text
            "embedding": embedding  # computed vector
        }
        batch.append(row)  # add to batch
        
        # When batch is full, upload it all at once
        if len(batch) == batch_size:
            print(f"📦 Sending a bulk batch of {len(batch)} items to Supabase Cloud...")
            supabase.table("documents").insert(batch).execute()  # bulk insert
            batch = [] # Clear batch

    # Upload remaining items
    if batch:
        print(f"📦 Sending final batch of {len(batch)} items to Supabase Cloud...")
        supabase.table("documents").insert(batch).execute()  # final insert

    print(f"✅ Successfully processed and synchronized {json_path} with Supabase!\n")

if __name__ == "__main__":
    # Since train likely uploaded mostly or fully, we can safely just run it again 
    # (or uncomment train if you want to be 100% sure nothing was left out)
    upload_dataset_to_supabase("dataset/qa_pairs_train.json")  # upload training set
    upload_dataset_to_supabase("dataset/qa_pairs_test.json")  # upload test set