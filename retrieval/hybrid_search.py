import os
import chromadb
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from embeddings.embedder import get_embeddings_batch
from embeddings.chunker import load_and_chunk_notes

load_dotenv()

class HybridSearch:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="data/vectorstore")
        self.collection = self.chroma_client.get_collection("clinical_notes")
        self.chunks = None
        self.bm25 = None
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from all chunks in ChromaDB"""
        print("Building BM25 index...")
        
        # Get all documents from ChromaDB
        results = self.collection.get(include=['documents', 'metadatas'])
        
        self.all_ids = results['ids']
        self.all_docs = results['documents']
        self.all_metadatas = results['metadatas']
        
        # Tokenize for BM25
        tokenized_docs = [doc.lower().split() for doc in self.all_docs]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        print(f"✅ BM25 index built with {len(self.all_docs)} documents")
    
    def semantic_search(self, query, n_results=10):
        """Pure semantic search using ChromaDB"""
        query_embedding = get_embeddings_batch([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return {
            'ids': results['ids'][0],
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
    
    def bm25_search(self, query, n_results=10):
        """BM25 keyword search"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top n results
        top_indices = sorted(range(len(scores)), 
                           key=lambda i: scores[i], 
                           reverse=True)[:n_results]
        
        return {
            'ids': [self.all_ids[i] for i in top_indices],
            'documents': [self.all_docs[i] for i in top_indices],
            'metadatas': [self.all_metadatas[i] for i in top_indices],
            'scores': [scores[i] for i in top_indices]
        }
    
    def reciprocal_rank_fusion(self, semantic_results, bm25_results, k=60):
        """Combine semantic and BM25 results using RRF"""
        rrf_scores = {}
        
        # Score semantic results
        for rank, doc_id in enumerate(semantic_results['ids']):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1 / (k + rank + 1)
        
        # Score BM25 results
        for rank, doc_id in enumerate(bm25_results['ids']):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1 / (k + rank + 1)
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), 
                          key=lambda x: rrf_scores[x], 
                          reverse=True)
        
        return sorted_ids, rrf_scores
    
    def hybrid_search(self, query, n_results=5):
        """Full hybrid search combining semantic + BM25 + RRF"""
        # Get results from both methods
        semantic_results = self.semantic_search(query, n_results=10)
        bm25_results = self.bm25_search(query, n_results=10)
        
        # Combine with RRF
        sorted_ids, rrf_scores = self.reciprocal_rank_fusion(
            semantic_results, bm25_results
        )
        
        # Get top n results
        top_ids = sorted_ids[:n_results]
        
        # Fetch full documents for top results
        final_results = []
        for doc_id in top_ids:
            # Find in semantic results first
            if doc_id in semantic_results['ids']:
                idx = semantic_results['ids'].index(doc_id)
                final_results.append({
                    'id': doc_id,
                    'document': semantic_results['documents'][idx],
                    'metadata': semantic_results['metadatas'][idx],
                    'rrf_score': rrf_scores[doc_id]
                })
            # Then check BM25 results
            elif doc_id in bm25_results['ids']:
                idx = bm25_results['ids'].index(doc_id)
                final_results.append({
                    'id': doc_id,
                    'document': bm25_results['documents'][idx],
                    'metadata': bm25_results['metadatas'][idx],
                    'rrf_score': rrf_scores[doc_id]
                })
        
        return final_results

def print_results(results, query):
    print(f"\n🔍 Query: '{query}'")
    print(f"Top {len(results)} results:")
    print("-" * 60)
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Specialty: {result['metadata']['specialty']}")
        print(f"  Sample: {result['metadata']['sample_name']}")
        print(f"  RRF Score: {result['rrf_score']:.4f}")
        print(f"  Text preview: {result['document'][:200]}...")

if __name__ == "__main__":
    print("=== Initializing Hybrid Search ===")
    searcher = HybridSearch()
    
    # Test queries
    queries = [
        "chest pain and heart attack treatment",
        "knee replacement surgery recovery",
        "diabetes medication management"
    ]
    
    for query in queries:
        results = searcher.hybrid_search(query, n_results=3)
        print_results(results, query)