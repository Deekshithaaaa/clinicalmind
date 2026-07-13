import os
from dotenv import load_dotenv
import psycopg2
from llm.summarizer import summarize_clinical_note
from llm.ner_extractor import extract_entities, extract_medical_keywords

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        database=os.getenv("POSTGRES_DB", "clinicalmind"),
        user=os.getenv("POSTGRES_USER", "clinicalmind"),
        password=os.getenv("POSTGRES_PASSWORD", "clinicalmind123")
    )

def full_analysis(note_id, specialty, sample_name, transcription):
    """Run full analysis: NER + Summarization on a clinical note"""
    
    print(f"\n{'='*70}")
    print(f"📋 {sample_name}")
    print(f"🏥 {specialty}")
    print(f"{'='*70}")
    
    # Step 1: NER extraction
    print("\n🔬 Step 1: Extracting Medical Entities...")
    entities = extract_entities(transcription)
    keywords = extract_medical_keywords(transcription)
    
    print(f"  Found {sum(len(v) for v in entities.values())} entities")
    if keywords['dosages']:
        print(f"  Dosages: {keywords['dosages'][:3]}")
    if keywords['diagnoses']:
        print(f"  Diagnoses: {keywords['diagnoses'][:3]}")
    if keywords['procedures']:
        print(f"  Procedures: {keywords['procedures'][:3]}")
    
    # Step 2: LLM Summarization
    print("\n📝 Step 2: Generating Clinical Summary...")
    summary = summarize_clinical_note(transcription, specialty)
    print(summary)
    
    return {
        'note_id': note_id,
        'specialty': specialty,
        'sample_name': sample_name,
        'entities': entities,
        'keywords': keywords,
        'summary': summary
    }

def run_full_pipeline(limit=3):
    """Run full NER + summarization pipeline"""
    print("=== ClinicalMind Full Analysis Pipeline ===\n")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, medical_specialty, sample_name, transcription
        FROM clean.clinical_notes
        WHERE word_count BETWEEN 150 AND 400
        ORDER BY RANDOM()
        LIMIT %s
    """, (limit,))
    
    notes = cursor.fetchall()
    cursor.close()
    conn.close()
    
    results = []
    for note in notes:
        note_id, specialty, sample_name, transcription = note
        result = full_analysis(note_id, specialty, sample_name, transcription)
        results.append(result)
    
    print(f"\n✅ Full pipeline completed for {len(results)} notes!")
    return results

if __name__ == "__main__":
    run_full_pipeline(limit=3)
