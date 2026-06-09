import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        database=os.getenv("POSTGRES_DB", "clinicalmind"),
        user=os.getenv("POSTGRES_USER", "clinicalmind"),
        password=os.getenv("POSTGRES_PASSWORD", "clinicalmind123")
    )

def chunk_text(text, chunk_size=800, overlap=150):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
        
        if start >= len(words):
            break
    
    return chunks

def load_and_chunk_notes(limit=500):
    """Load clinical notes from PostgreSQL and chunk them"""
    print("=== Loading and Chunking Clinical Notes ===")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Load notes from clean layer
    cursor.execute("""
        SELECT 
            id,
            medical_specialty,
            sample_name,
            transcription,
            specialty_category,
            word_count
        FROM clean.clinical_notes
        WHERE word_count > 100
        ORDER BY id
        LIMIT %s
    """, (limit,))
    
    notes = cursor.fetchall()
    print(f"Loaded {len(notes)} notes from PostgreSQL")
    
    # Chunk each note
    all_chunks = []
    for note in notes:
        note_id, specialty, sample_name, transcription, category, word_count = note
        
        chunks = chunk_text(transcription, chunk_size=800, overlap=150)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'id': f"note_{note_id}_chunk_{i}",
                'text': chunk,
                'metadata': {
                    'note_id': note_id,
                    'specialty': specialty,
                    'sample_name': sample_name,
                    'category': category,
                    'word_count': word_count,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            })
    
    cursor.close()
    conn.close()
    
    print(f"Created {len(all_chunks)} chunks from {len(notes)} notes")
    print(f"Average chunks per note: {len(all_chunks)/len(notes):.1f}")
    
    return all_chunks

if __name__ == "__main__":
    chunks = load_and_chunk_notes(limit=500)
    print(f"\nSample chunk:")
    print(f"ID: {chunks[0]['id']}")
    print(f"Specialty: {chunks[0]['metadata']['specialty']}")
    print(f"Text preview: {chunks[0]['text'][:200]}...")