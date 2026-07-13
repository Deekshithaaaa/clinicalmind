import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from datasets import Dataset
import chromadb
from rank_bm25 import BM25Okapi
from embeddings.embedder import get_embeddings_batch
from evaluation.test_questions import TEST_QUESTIONS

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    emb = get_embeddings_batch([query])[0]
    sem = collection.query(query_embeddings=[emb], n_results=10)
    sem_ids = sem["ids"][0]
    scores = bm25.get_scores(query.lower().split())
    top_bm25 = sorted(range(len(scores)),
                      key=lambda i: scores[i], reverse=True)[:10]
    bm25_ids = [all_data["ids"][i] for i in top_bm25]
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

def generate_answer(question, chunks):
    context = "\n\n".join(
        f"Source {i+1} ({c['metadata']['specialty']}): {c['document'][:500]}"
        for i, c in enumerate(chunks)
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content":
             "You are a clinical assistant. Answer based only on the "
             "provided context. Be precise and cite sources."},
            {"role": "user", "content":
             f"Question: {question}\n\nContext:\n{context}"}
        ],
        max_tokens=300,
        temperature=0.1
    )
    return response.choices[0].message.content

def build_ragas_dataset(search_fn, collection, n_questions=15):
    """Build dataset for RAGAs evaluation"""
    questions, answers, contexts, ground_truths = [], [], [], []

    for i, item in enumerate(TEST_QUESTIONS[:n_questions]):
        question = item["question"]
        print(f"  [{i+1:02d}/{n_questions}] {question[:55]}...")

        chunks = search_fn(question, collection)
        answer = generate_answer(question, chunks)

        questions.append(question)
        answers.append(answer)
        contexts.append([c["document"][:500] for c in chunks])
        ground_truths.append(question)  # Using question as proxy ground truth

        time.sleep(0.3)

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

def run_ragas_evaluation():
    print("=== RAGAs Evaluation Pipeline ===\n")

    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy

    collection = get_collection()

    # Evaluate semantic search
    print("📊 Evaluating Semantic Search...")
    sem_dataset = build_ragas_dataset(semantic_search, collection, n_questions=15)
    sem_results = evaluate(
        sem_dataset,
        metrics=[faithfulness, answer_relevancy]
    )
    sem_df = sem_results.to_pandas()

    print(f"\n✅ Semantic Search Results:")
    print(f"   Faithfulness:     {sem_df['faithfulness'].mean():.3f}")
    print(f"   Answer Relevancy: {sem_df['answer_relevancy'].mean():.3f}")

    # Evaluate hybrid search
    print("\n📊 Evaluating Hybrid Search...")
    hyb_dataset = build_ragas_dataset(hybrid_search, collection, n_questions=15)
    hyb_results = evaluate(
        hyb_dataset,
        metrics=[faithfulness, answer_relevancy]
    )
    hyb_df = hyb_results.to_pandas()

    print(f"\n✅ Hybrid Search Results:")
    print(f"   Faithfulness:     {hyb_df['faithfulness'].mean():.3f}")
    print(f"   Answer Relevancy: {hyb_df['answer_relevancy'].mean():.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("         RAGAS EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'Semantic':>10} {'Hybrid':>10} {'Winner':>10}")
    print("-" * 60)

    sem_faith = sem_df['faithfulness'].mean()
    hyb_faith = hyb_df['faithfulness'].mean()
    faith_winner = "Hybrid ✅" if hyb_faith > sem_faith else "Semantic ✅"

    sem_rel = sem_df['answer_relevancy'].mean()
    hyb_rel = hyb_df['answer_relevancy'].mean()
    rel_winner = "Hybrid ✅" if hyb_rel > sem_rel else "Semantic ✅"

    print(f"{'Faithfulness':<25} {sem_faith:>10.3f} {hyb_faith:>10.3f} {faith_winner:>10}")
    print(f"{'Answer Relevancy':<25} {sem_rel:>10.3f} {hyb_rel:>10.3f} {rel_winner:>10}")
    print("=" * 60)

    # Save results
    results = {
        "semantic": {
            "faithfulness": float(sem_faith),
            "answer_relevancy": float(sem_rel)
        },
        "hybrid": {
            "faithfulness": float(hyb_faith),
            "answer_relevancy": float(hyb_rel)
        }
    }

    with open("evaluation/ragas_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to evaluation/ragas_results.json")
    return results

if __name__ == "__main__":
    run_ragas_evaluation()
