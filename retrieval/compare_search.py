import os
import chromadb
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from embeddings.embedder import get_embeddings_batch

load_dotenv()

def compare_search_methods(query, n_results=3):
    """Compare semantic vs hybrid search results"""
    
    client = chromadb.PersistentClient(path="data/vectorstore")
    collection = client.get_collection("clinical_notes")
    
    # Get all docs for BM25
    all_data = collection.get(include=['documents', 'metadatas'])
    all_ids = all_data['ids']
    all_docs = all_data['documents']
    all_metadatas = all_data['metadatas']
    
    # Build BM25
    tokenized_docs = [doc.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_docs)
    
    # --- Semantic Search ---
    query_embedding = get_embeddings_batch([query])[0]
    semantic = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    # --- BM25 Search ---
    scores = bm25.get_scores(query.lower().split())
    top_bm25 = sorted(range(len(scores)), 
                      key=lambda i: scores[i], reverse=True)[:n_results]
    
    # --- Print Comparison ---
    print(f"\n{'='*70}")
    print(f"Query: '{query}'")
    print(f"{'='*70}")
    
    print(f"\n{'SEMANTIC SEARCH':^35} | {'BM25 KEYWORD SEARCH':^35}")
    print(f"{'-'*35}-+-{'-'*35}")
    
    for i in range(n_results):
        sem_sample = semantic['metadatas'][0][i]['sample_name'][:30]
        sem_spec = semantic['metadatas'][0][i]['specialty'][:20]
        sem_score = f"{1 - semantic['distances'][0][i]:.3f}"
        
        bm25_idx = top_bm25[i]
        bm25_sample = all_metadatas[bm25_idx]['sample_name'][:30]
        bm25_spec = all_metadatas[bm25_idx]['specialty'][:20]
        bm25_score = f"{scores[bm25_idx]:.3f}"
        
        print(f"{sem_sample:<35} | {bm25_sample:<35}")
        print(f"  {sem_spec:<33} | {bm25_spec:<33}")
        print(f"  Score: {sem_score:<28} | Score: {bm25_score:<28}")
        print(f"{'-'*35}-+-{'-'*35}")

if __name__ == "__main__":
    queries = [
        "chest pain and heart attack treatment",
        "knee replacement surgery recovery",
        "diabetes medication management"
    ]
    
    for query in queries:
        compare_search_methods(query, n_results=3)