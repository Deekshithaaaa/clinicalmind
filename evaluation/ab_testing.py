import os
import json
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from rank_bm25 import BM25Okapi
from embeddings.embedder import get_embeddings_batch
from evaluation.test_questions import TEST_QUESTIONS

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Search helpers ──────────────────────────────────────────
def get_collection():
    c = chromadb.PersistentClient(path="data/vectorstore")
    return c.get_collection("clinical_notes")

def semantic_search(query, collection, n=5):
    emb = get_embeddings_batch([query])[0]
    r = collection.query(query_embeddings=[emb], n_results=n)
    return [{"document": d, "metadata": m}
            for d, m in zip(r["documents"][0], r["metadatas"][0])]

def hybrid_search(query, collection, n=5):
    all_data = collection.get(include=["documents", "metadatas"])
    tokenized = [d.lower().split() for d in all_data["documents"]]
    bm25 = BM25Okapi(tokenized)

    # semantic
    emb = get_embeddings_batch([query])[0]
    sem = collection.query(query_embeddings=[emb], n_results=10)
    sem_ids = sem["ids"][0]

    # bm25
    scores = bm25.get_scores(query.lower().split())
    top_bm25 = sorted(range(len(scores)),
                      key=lambda i: scores[i], reverse=True)[:10]
    bm25_ids = [all_data["ids"][i] for i in top_bm25]

    # RRF
    rrf = {}
    for rank, did in enumerate(sem_ids):
        rrf[did] = rrf.get(did, 0) + 1 / (60 + rank + 1)
    for rank, did in enumerate(bm25_ids):
        rrf[did] = rrf.get(did, 0) + 1 / (60 + rank + 1)

    top_ids = sorted(rrf, key=rrf.get, reverse=True)[:n]

    results = []
    for did in top_ids:
        if did in sem_ids:
            idx = sem_ids.index(did)
            results.append({
                "document": sem["documents"][0][idx],
                "metadata": sem["metadatas"][0][idx]
            })
        else:
            idx = all_data["ids"].index(did)
            results.append({
                "document": all_data["documents"][idx],
                "metadata": all_data["metadatas"][idx]
            })
    return results

# ── Answer generation ────────────────────────────────────────
def generate_answer(question, chunks):
    context = "\n\n".join(
        f"Source {i+1} ({c['metadata']['specialty']}): {c['document'][:500]}"
        for i, c in enumerate(chunks)
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             "You are a clinical assistant. Answer based only on the provided context. "
             "Be precise and cite sources."},
            {"role": "user", "content": 
             f"Question: {question}\n\nContext:\n{context}"}
        ],
        max_tokens=300,
        temperature=0.1
    )
    return response.choices[0].message.content

# ── Faithfulness scoring ─────────────────────────────────────
def score_faithfulness(question, answer, chunks):
    context = "\n\n".join(c["document"][:300] for c in chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content":
             "Score how faithful the answer is to the context on a scale of 0.0 to 1.0. "
             "1.0 = completely faithful, 0.0 = completely unfaithful or hallucinated. "
             "Respond with ONLY a number between 0.0 and 1.0."},
            {"role": "user", "content":
             f"Question: {question}\nAnswer: {answer}\nContext: {context}"}
        ],
        max_tokens=10,
        temperature=0
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.5

# ── Main experiment ──────────────────────────────────────────
def run_ab_test(n_questions=25):
    print("=== ClinicalMind A/B Test ===")
    print(f"Testing {n_questions} questions")
    print("Control:   Semantic Search")
    print("Treatment: Hybrid Search (BM25 + Semantic + RRF)\n")

    collection = get_collection()
    semantic_scores = []
    hybrid_scores   = []

    for i, item in enumerate(TEST_QUESTIONS[:n_questions]):
        question = item["question"]
        print(f"[{i+1:02d}/{n_questions}] {question[:60]}...")

        # Control – semantic
        sem_chunks  = semantic_search(question, collection)
        sem_answer  = generate_answer(question, sem_chunks)
        sem_score   = score_faithfulness(question, sem_answer, sem_chunks)
        semantic_scores.append(sem_score)

        # Treatment – hybrid
        hyb_chunks  = hybrid_search(question, collection)
        hyb_answer  = generate_answer(question, hyb_chunks)
        hyb_score   = score_faithfulness(question, hyb_answer, hyb_chunks)
        hybrid_scores.append(hyb_score)

        print(f"    Semantic: {sem_score:.3f}  |  Hybrid: {hyb_score:.3f}")
        time.sleep(0.5)

    return semantic_scores, hybrid_scores

# ── Statistical analysis ─────────────────────────────────────
def analyze_results(semantic_scores, hybrid_scores):
    sem_arr = np.array(semantic_scores)
    hyb_arr = np.array(hybrid_scores)

    stat, p_value = stats.mannwhitneyu(
        hyb_arr, sem_arr, alternative="greater"
    )

    improvement = ((hyb_arr.mean() - sem_arr.mean()) / sem_arr.mean()) * 100

    print("\n" + "=" * 60)
    print("         A/B TEST RESULTS")
    print("=" * 60)
    print(f"Semantic Search  — avg faithfulness: {sem_arr.mean():.3f} "
          f"(±{sem_arr.std():.3f})")
    print(f"Hybrid Search    — avg faithfulness: {hyb_arr.mean():.3f} "
          f"(±{hyb_arr.std():.3f})")
    print(f"Improvement:      {improvement:+.1f}%")
    print(f"Mann-Whitney U:   {stat:.1f}")
    print(f"p-value:          {p_value:.4f}")
    print(f"Significant:      {'✅ YES (p < 0.05)' if p_value < 0.05 else '❌ NO (p >= 0.05)'}")
    print("=" * 60)

    # Chart
    os.makedirs("evaluation", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].boxplot([semantic_scores, hybrid_scores],
                    labels=["Semantic", "Hybrid"])
    axes[0].set_title("Faithfulness Score Distribution")
    axes[0].set_ylabel("Faithfulness Score")
    axes[0].set_ylim(0, 1.1)

    x = range(1, len(semantic_scores) + 1)
    axes[1].plot(x, semantic_scores, "b-o", label="Semantic",
                 alpha=0.7, markersize=4)
    axes[1].plot(x, hybrid_scores,   "r-o", label="Hybrid",
                 alpha=0.7, markersize=4)
    axes[1].axhline(sem_arr.mean(), color="blue",  linestyle="--", alpha=0.5)
    axes[1].axhline(hyb_arr.mean(), color="red",   linestyle="--", alpha=0.5)
    axes[1].set_title("Faithfulness Scores per Question")
    axes[1].set_xlabel("Question Number")
    axes[1].set_ylabel("Faithfulness Score")
    axes[1].legend()

    plt.tight_layout()
    chart_path = os.path.join(
        os.path.expanduser("~"), "clinicalmind",
        "evaluation", "ab_test_results.png"
    )
    plt.savefig(chart_path, dpi=150)
    plt.show()
    print(f"\n📊 Chart saved to: {chart_path}")

    # Save raw results
    results = {
        "semantic_scores": semantic_scores,
        "hybrid_scores": hybrid_scores,
        "semantic_mean": float(sem_arr.mean()),
        "hybrid_mean": float(hyb_arr.mean()),
        "improvement_pct": float(improvement),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05)
    }
    json_path = os.path.join(
        os.path.expanduser("~"), "clinicalmind",
        "evaluation", "ab_test_results.json"
    )
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to: {json_path}")

    return results

if __name__ == "__main__":
    semantic_scores, hybrid_scores = run_ab_test(n_questions=25)
    results = analyze_results(semantic_scores, hybrid_scores)
