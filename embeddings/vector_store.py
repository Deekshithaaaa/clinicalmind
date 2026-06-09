import chromadb
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

VECTORSTORE_PATH = os.path.join(os.path.expanduser('~'), 'clinicalmind', 'data', 'vectorstore')

def get_chroma_client():
    return chromadb.PersistentClient(path=VECTORSTORE_PATH)

def create_vector_store(embedded_chunks):
    """Store embeddings in ChromaDB"""
    print("=== Creating ChromaDB Vector Store ===")
    
    client = get_chroma_client()
    
    # Delete existing collection if exists
    try:
        client.delete_collection("clinical_notes")
        print("Deleted existing collection")
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name="clinical_notes",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add chunks in batches
    batch_size = 100
    total_added = 0
    
    for i in tqdm(range(0, len(embedded_chunks), batch_size), desc="Adding to ChromaDB"):
        batch = embedded_chunks[i:i + batch_size]
        
        ids = [chunk['id'] for chunk in batch]
        embeddings = [chunk['embedding'] for chunk in batch]
        documents = [chunk['text'] for chunk in batch]
        metadatas = [chunk['metadata'] for chunk in batch]
        
        # Convert metadata values to strings/ints (ChromaDB requirement)
        cleaned_metadatas = []
        for meta in metadatas:
            cleaned_meta = {k: str(v) if not isinstance(v, (int, float)) else v 
                          for k, v in meta.items()}
            cleaned_metadatas.append(cleaned_meta)
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=cleaned_metadatas
        )
        total_added += len(batch)
    
    print(f"\n✅ Added {total_added} chunks to ChromaDB")
    print(f"Collection: clinical_notes")
    print(f"Storage path: {VECTORSTORE_PATH}")
    
    # Verify
    count = collection.count()
    print(f"Verified count: {count} chunks in vector store")
    
    return collection

def test_semantic_search(query, n_results=3):
    """Test semantic search"""
    from embedder import generate_embeddings
    
    print(f"\n=== Testing Semantic Search ===")
    print(f"Query: '{query}'")
    
    # Embed the query
    query_chunk = [{'id': 'query', 'text': query, 'metadata': {}}]
    embedded_query = generate_embeddings(query_chunk, batch_size=1)
    query_embedding = embedded_query[0]['embedding']
    
    # Search
    client = get_chroma_client()
    collection = client.get_collection("clinical_notes")
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    
    print(f"\nTop {n_results} results:")
    for i in range(len(results['documents'][0])):
        print(f"\n--- Result {i+1} ---")
        print(f"Specialty: {results['metadatas'][0][i]['specialty']}")
        print(f"Distance: {results['distances'][0][i]:.4f}")
        print(f"Text preview: {results['documents'][0][i][:200]}...")
    
    return results

if __name__ == "__main__":
    from chunker import load_and_chunk_notes
    from embedder import generate_embeddings
    
    # Load, chunk and embed
    chunks = load_and_chunk_notes(limit=500)
    embedded_chunks = generate_embeddings(chunks, batch_size=50)
    
    # Store in ChromaDB
    collection = create_vector_store(embedded_chunks)
    
    # Test semantic search
    test_semantic_search("chest pain and heart attack treatment")
    test_semantic_search("knee surgery recovery")
    test_semantic_search("diabetes medication management")
