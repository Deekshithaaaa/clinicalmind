import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_embeddings(chunks, batch_size=50):
    """Generate embeddings for all chunks using OpenAI"""
    print("=== Generating Embeddings ===")
    print(f"Total chunks to embed: {len(chunks)}")
    print(f"Model: text-embedding-3-small")
    
    embedded_chunks = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i:i + batch_size]
        texts = [chunk['text'] for chunk in batch]
        
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            
            for j, embedding_data in enumerate(response.data):
                chunk = batch[j].copy()
                chunk['embedding'] = embedding_data.embedding
                embedded_chunks.append(chunk)
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error embedding batch {i}: {e}")
            time.sleep(2)
            continue
    
    print(f"\n✅ Generated {len(embedded_chunks)} embeddings")
    print(f"Embedding dimensions: {len(embedded_chunks[0]['embedding'])}")
    return embedded_chunks

if __name__ == "__main__":
    from chunker import load_and_chunk_notes
    
    chunks = load_and_chunk_notes(limit=500)
    embedded_chunks = generate_embeddings(chunks, batch_size=50)
    
    print(f"\nSample embedding:")
    print(f"ID: {embedded_chunks[0]['id']}")
    print(f"Embedding length: {len(embedded_chunks[0]['embedding'])}")
    print(f"First 5 values: {embedded_chunks[0]['embedding'][:5]}")
