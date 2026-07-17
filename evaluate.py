import os  # os utilities
import json  # json parsing
import requests  # HTTP calls for Jina reranker
import numpy as np  # numerical operations
import pandas as pd  # dataframes
from evaluation_metrics import (
    compute_context_precision,
    compute_context_recall,
    compute_hit_at_1,
    compute_hit_at_3,
    compute_hit_at_5,
)
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # visualization styles
import time  # sleep and timing
from dotenv import load_dotenv  # load .env
from supabase import create_client, Client  # supabase client
from llama_index.llms.groq import Groq  # Groq LLM wrapper
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # embedding model
from scipy.spatial.distance import cosine  # cosine similarity
from sklearn.metrics import confusion_matrix, classification_report  # evaluation metrics

# Load environment variables

load_dotenv()  # load .env into environment

# Project-local HuggingFace cache directory to avoid global side-effects and deprecation warnings
CACHE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".hf_cache"))
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize Supabase, Embedding, and Groq
supabase_url = os.getenv("SUPABASE_URL")  # supabase URL
supabase_key = os.getenv("SUPABASE_KEY")  # supabase key
supabase: Client = create_client(supabase_url, supabase_key)  # init supabase client

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"cache_dir": CACHE_DIR}
)  # embedding model instance with project cache
llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))  # LLM instance


def rerank_context_with_jina(query, retrieved_chapters):
    """Rerank the top retrieved chunks using Jina API and preserve source metadata."""
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
        print(f"Jina reranker failed: {e}. Falling back to raw similarity order.")
        fallback = []
        for sim, row, _ in retrieved_chapters[:3]:
            fallback.append({
                "text": row.get("text", ""),
                "source": row.get("source"),
                "page": row.get("page"),
                "similarity": sim,
            })
        return fallback


def run_rag_pipeline(user_input):
    """Run the current Supabase retrieval + Jina reranking RAG pipeline."""
    try:
        query_embedding = embed_model.get_text_embedding(user_input)
        response_db = supabase.table("documents").select("*, embedding").execute()

        candidates = []
        for row in response_db.data:
            db_emb = row.get("embedding")
            if isinstance(db_emb, str):
                db_emb = json.loads(db_emb)
            if not db_emb:
                continue
            sim = 1 - cosine(query_embedding, db_emb)
            candidates.append((sim, row, db_emb))

        if not candidates:
            return "❌ No confident match found.", "", 0.0, [], ""

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
        ai_answer = response_ai.text

        metadata_text = (
            f"📚 {prompt_sources} | Confidence: {top_confidence:.2%}"
            if prompt_sources
            else f"Confidence: {top_confidence:.2%}"
        )

        return ai_answer, context_string, top_confidence, metadata_text, reranked_items

    except Exception as e:
        print(f"Error in pipeline: {e}")
        return "", "", 0.0, "", []


def evaluate_faithfulness(generated_answer, context):
    """Menggunakan LLM Groq sebagai hakim untuk mengira skor Faithfulness (0.0 - 1.0)"""  # docstring
    if not context or "No confident match" in generated_answer:  # quick checks
        return 0.0  # no context -> not faithful
    
    prompt = f"""
    Given the following context and a generated answer, judge if the generated answer is faithful to the context.
    Does the generated answer contain any hallucinations or information NOT present in the context?
    
    CONTEXT: {context}
    GENERATED ANSWER: {generated_answer}
    
    Output ONLY a single float number between 0.0 (completely hallucinated) and 1.0 (completely faithful/honest).
    Do not include any explanation or extra text. Just the number.
    """  # prompt for LLM judge
    try:
        res = llm.complete(prompt).text.strip()  # call LLM
        return float(res)  # parse float
    except:
        return 1.0 # Default if parsing fails

def evaluate_answer_relevance(generated_answer, user_question):
    """Menggunakan LLM Groq sebagai hakim untuk mengira skor Answer Relevance (0.0 - 1.0)"""  # docstring
    if "No confident match" in generated_answer:  # check fallback
        return 0.0
        
    prompt = f"""
    Given the user's question and the generated answer, judge how relevant the answer is to the question.
    Does it directly address what the user asked?
    
    USER QUESTION: {user_question}
    GENERATED ANSWER: {generated_answer}
    
    Output ONLY a single float number between 0.0 (completely irrelevant) and 1.0 (highly relevant and direct).
    Do not include any explanation or extra text. Just the number.
    """  # prompt for relevance judge
    try:
        res = llm.complete(prompt).text.strip()  # call LLM
        return float(res)  # parse result
    except:
        return 1.0  # default

# --- MAIN EVALUATION PROCESS ---
dataset_path = os.getenv("EVAL_DATASET_PATH", "dataset/qa_pairs_test_new.json")
print(f"📖 Loading test dataset from {dataset_path}...")  # log
with open(dataset_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)  # load test data

# Batasi jumlah test jika fail terlalu besar untuk elak hit Groq Rate Limit (Contoh: ambil 20 soalan terawal)
# Kalau nak uji semua sekali, buang [:20] ini.
test_subset = test_data[:20]  # limit tests to first 20

results = []  # collect evaluation results

print(f"🚀 Running evaluation on {len(test_subset)} test queries...")  # progress log
for idx, item in enumerate(test_subset):  # iterate test items
    question = item["question"]  # user question from dataset
    ground_truth = item["answer"]  # expected answer
    
    print(f"[{idx+1}/{len(test_subset)}] Evaluating: {question[:40]}...")  # per-item log
    
    # Jalankan pipeline RAG Live
    ai_answer, retrieved_context, retrieval_score, metadata_text, reranked_items = run_rag_pipeline(question)  # get AI response
    
    # Bagi Groq bernafas sekejap (Rate limit mitigation)
    time.sleep(2)  # short delay
    
    # Nilai kualiti menggunakan LLM as a Judge
    faithfulness = evaluate_faithfulness(ai_answer, retrieved_context)  # faithfulness score
    time.sleep(2)  # pause between calls
    
    relevance = evaluate_answer_relevance(ai_answer, question)  # relevance score
    time.sleep(2) # Delay sebelum beralih ke soalan seterusnya
    
    hit_at_1 = compute_hit_at_1(reranked_items, question, ground_truth)
    hit_at_3 = compute_hit_at_3(reranked_items, question, ground_truth)
    hit_at_5 = compute_hit_at_5(reranked_items, question, ground_truth)
    context_precision = compute_context_precision(reranked_items, question, ground_truth)
    context_recall = compute_context_recall(reranked_items, question, ground_truth)

    results.append({
        "Question": question,
        "Ground Truth": ground_truth,
        "AI Answer": ai_answer,
        "Metadata": metadata_text,
        "Retrieval Score": retrieval_score,
        "Faithfulness": faithfulness,
        "Answer Relevance": relevance,
        "Hit@1": hit_at_1,
        "Hit@3": hit_at_3,
        "Hit@5": hit_at_5,
        "Context Precision": context_precision,
        "Context Recall": context_recall,
        "Answer Relevancy": relevance
    })  # append result dict

# Convert to DataFrame
df = pd.DataFrame(results)  # create dataframe from results

# --- 2. GENERATE CHARTS ---
print("\n📊 Generating Evaluation Graphs...")  # log
os.makedirs("evaluation_results", exist_ok=True)  # ensure output dir

# Graph 1: Average Scores Bar Chart
plt.figure(figsize=(8, 5))  # create figure
avg_scores = [df["Retrieval Score"].mean(), df["Faithfulness"].mean(), df["Answer Relevance"].mean()]  # compute averages
metrics = ["Context Retrieval", "Faithfulness", "Answer Relevance"]  # metric names
sns.barplot(x=metrics, y=avg_scores, palette="viridis")  # draw barplot
plt.ylim(0, 1.1)  # set y limits
plt.title("Average RAG Evaluation Scores")  # title
plt.ylabel("Score (0.0 - 1.0)")  # y label
for i, v in enumerate(avg_scores):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')  # annotate values
plt.savefig("evaluation_results/average_scores.png")  # save chart
plt.close()  # close figure

# Graph 2: Confusion Matrix Heatmap
# Kita takrifkan: Skor Faithfulness >= 0.75 adalah "Faithful/Jujur" (1), bawah 0.75 adalah "Hallucinated/Fakta Rekaan" (0)
# Kita bandingkan dengan keadaan ideal (Ground Truth assume semua soalan test patutnya 100% Faithful = 1)
df["Actual_Faithful"] = 1  # Baseline ideal
df["Predicted_Faithful"] = np.where(df["Faithfulness"] >= 0.75, 1, 0)  # predicted label

# Jika ada kes kosong (AI skor rendah semua), kita pastikan bentuk matrix kekal 2x2
cm = confusion_matrix(df["Actual_Faithful"], df["Predicted_Faithful"], labels=[0, 1])  # compute confusion matrix

plt.figure(figsize=(6, 5))  # new figure
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Hallucinated", "Faithful"], 
            yticklabels=["Hallucinated", "Faithful"])  # plot heatmap
plt.title("RAG Faithfulness Confusion Matrix")  # title
plt.ylabel("Actual Baseline Quality")  # y label
plt.xlabel("Predicted Chatbot Quality")  # x label
plt.savefig("evaluation_results/confusion_matrix.png")  # save figure
plt.close()  # close figure

# Save CSV Report
df.to_csv("evaluation_results/detailed_report.csv", index=False)  # save csv

print("\n✨ Evaluation Complete!")  # final log
print(f"📈 Average Retrieval (Cosine Sim): {df['Retrieval Score'].mean():.2f}")  # avg retrieval
print(f"📈 Average Faithfulness: {df['Faithfulness'].mean():.2f}")  # avg faithfulness
print(f"📈 Average Answer Relevance: {df['Answer Relevance'].mean():.2f}")  # avg relevance
print(f"📈 Average Hit@1: {df['Hit@1'].mean():.2f}")  # avg hit@1
print(f"📈 Average Hit@3: {df['Hit@3'].mean():.2f}")  # avg hit@3
print(f"📈 Average Hit@5: {df['Hit@5'].mean():.2f}")  # avg hit@5
print(f"📈 Average Context Precision: {df['Context Precision'].mean():.2f}")  # avg context precision
print(f"📈 Average Context Recall: {df['Context Recall'].mean():.2f}")  # avg context recall
print("📁 All charts and CSV reports have been successfully saved inside the 'evaluation_results/' folder!")  # finished