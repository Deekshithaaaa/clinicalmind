import os
import spacy
import psycopg2
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

nlp = spacy.load("en_core_sci_sm")

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        database=os.getenv("POSTGRES_DB", "clinicalmind"),
        user=os.getenv("POSTGRES_USER", "clinicalmind"),
        password=os.getenv("POSTGRES_PASSWORD", "clinicalmind123")
    )

def extract_entities(text):
    doc = nlp(text[:10000])
    entities = defaultdict(list)
    for ent in doc.ents:
        label = ent.label_
        text_val = ent.text.strip()
        if len(text_val) < 2 or len(text_val) > 100:
            continue
        if text_val.isdigit():
            continue
        entities[label].append(text_val)
    for label in entities:
        entities[label] = list(set(entities[label]))
    return dict(entities)

def extract_medical_keywords(text):
    import re
    keywords = {
        'medications': [],
        'dosages': [],
        'procedures': [],
        'diagnoses': []
    }
    dosage_pattern = r'\b\d+\.?\d*\s*(?:mg|mcg|ml|g|units?|tablets?|capsules?)\b'
    keywords['dosages'] = re.findall(dosage_pattern, text, re.IGNORECASE)
    procedure_keywords = ['surgery', 'procedure', 'operation', 'biopsy', 'catheterization',
        'angioplasty', 'replacement', 'repair', 'resection', 'excision',
        'laparoscopic', 'endoscopy', 'colonoscopy', 'angiogram']
    for keyword in procedure_keywords:
        if keyword.lower() in text.lower():
            keywords['procedures'].append(keyword)
    diagnosis_keywords = ['diabetes', 'hypertension', 'cancer', 'tumor', 'infection',
        'fracture', 'arthritis', 'pneumonia', 'asthma', 'COPD',
        'atrial fibrillation', 'heart failure', 'stroke', 'MI']
    for keyword in diagnosis_keywords:
        if keyword.lower() in text.lower():
            keywords['diagnoses'].append(keyword)
    return keywords

def analyze_note(note_id, specialty, sample_name, transcription):
    print(f"\n{'='*60}")
    print(f"📋 Note: {sample_name}")
    print(f"🏥 Specialty: {specialty}")
    print(f"{'='*60}")
    entities = extract_entities(transcription)
    keywords = extract_medical_keywords(transcription)
    print(f"\n🔬 scispaCy Entities Found:")
    if entities:
        for label, values in entities.items():
            print(f"  {label}: {values[:5]}")
    else:
        print("  No entities found")
    print(f"\n💊 Medical Keywords:")
    for category, values in keywords.items():
        if values:
            print(f"  {category}: {list(set(values))[:5]}")
    return {
        'note_id': note_id,
        'specialty': specialty,
        'sample_name': sample_name,
        'entities': entities,
        'keywords': keywords
    }

def run_ner_pipeline(limit=5):
    print("=== Medical NER Pipeline ===\n")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, medical_specialty, sample_name, transcription
        FROM clean.clinical_notes
        WHERE word_count BETWEEN 100 AND 400
        ORDER BY RANDOM()
        LIMIT %s
    """, (limit,))
    notes = cursor.fetchall()
    cursor.close()
    conn.close()
    results = []
    for note in notes:
        note_id, specialty, sample_name, transcription = note
        result = analyze_note(note_id, specialty, sample_name, transcription)
        results.append(result)
    print(f"\n✅ NER completed for {len(results)} clinical notes!")
    return results

if __name__ == "__main__":
    run_ner_pipeline(limit=5)
