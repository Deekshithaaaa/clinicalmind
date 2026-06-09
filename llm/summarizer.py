import os
from openai import OpenAI
from dotenv import load_dotenv
import psycopg2

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        database=os.getenv("POSTGRES_DB", "clinicalmind"),
        user=os.getenv("POSTGRES_USER", "clinicalmind"),
        password=os.getenv("POSTGRES_PASSWORD", "clinicalmind123")
    )

def summarize_clinical_note(transcription, specialty):
    """Summarize a clinical note into structured format"""
    
    system_prompt = """You are a clinical documentation specialist. 
Your task is to summarize clinical notes into a structured format.
Always extract exactly these sections if present:
- Chief Complaint
- Diagnosis
- Key Findings
- Medications
- Procedures
- Follow-up Plan

Be concise and precise. Use medical terminology appropriately.
If a section is not mentioned in the note, write 'Not mentioned'.
Never add information not present in the original note."""

    user_prompt = f"""Please summarize this {specialty} clinical note:

{transcription[:3000]}

Provide a structured summary with the sections listed."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=500,
        temperature=0.1
    )
    
    return response.choices[0].message.content

def summarize_batch(limit=5):
    """Summarize a batch of clinical notes"""
    print("=== Clinical Note Summarization ===\n")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, medical_specialty, sample_name, transcription
        FROM clean.clinical_notes
        WHERE word_count BETWEEN 100 AND 500
        ORDER BY RANDOM()
        LIMIT %s
    """, (limit,))
    
    notes = cursor.fetchall()
    cursor.close()
    conn.close()
    
    summaries = []
    
    for note_id, specialty, sample_name, transcription in notes:
        print(f"📋 Summarizing: {sample_name} ({specialty})")
        print("-" * 60)
        
        summary = summarize_clinical_note(transcription, specialty)
        
        print(summary)
        print("\n")
        
        summaries.append({
            'note_id': note_id,
            'specialty': specialty,
            'sample_name': sample_name,
            'original': transcription,
            'summary': summary
        })
    
    return summaries

if __name__ == "__main__":
    summaries = summarize_batch(limit=3)
    print(f"✅ Summarized {len(summaries)} clinical notes!")