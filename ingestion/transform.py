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

def transform_notes():
    print("=== Starting Transformations ===")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Clear existing clean data
    cursor.execute("TRUNCATE TABLE clean.clinical_notes RESTART IDENTITY CASCADE")
    cursor.execute("TRUNCATE TABLE clean.specialty_stats RESTART IDENTITY CASCADE")
    conn.commit()
    
    # Transform raw notes into clean layer
    transform_query = """
        INSERT INTO clean.clinical_notes 
        (raw_id, medical_specialty, sample_name, transcription, 
         word_count, char_count, specialty_category)
        SELECT 
            id as raw_id,
            TRIM(medical_specialty) as medical_specialty,
            TRIM(sample_name) as sample_name,
            TRIM(transcription) as transcription,
            array_length(
                string_to_array(TRIM(transcription), ' '), 1
            ) as word_count,
            LENGTH(TRIM(transcription)) as char_count,
            CASE 
                WHEN medical_specialty ILIKE '%surgery%' THEN 'Surgical'
                WHEN medical_specialty ILIKE '%cardio%' 
                  OR medical_specialty ILIKE '%pulmonary%' THEN 'Cardiac'
                WHEN medical_specialty ILIKE '%neuro%' THEN 'Neurological'
                WHEN medical_specialty ILIKE '%ortho%' THEN 'Orthopedic'
                WHEN medical_specialty ILIKE '%radiology%' THEN 'Radiology'
                WHEN medical_specialty ILIKE '%gastro%' THEN 'Gastroenterology'
                WHEN medical_specialty ILIKE '%obstet%' 
                  OR medical_specialty ILIKE '%gynec%' THEN 'OB/GYN'
                ELSE 'General'
            END as specialty_category
        FROM raw.clinical_notes
        WHERE transcription IS NOT NULL
        AND LENGTH(TRIM(transcription)) > 50
    """
    
    cursor.execute(transform_query)
    conn.commit()
    
    # Verify clean notes
    cursor.execute("SELECT COUNT(*) FROM clean.clinical_notes")
    clean_count = cursor.fetchone()[0]
    print(f"✅ Transformed {clean_count} records into clean.clinical_notes")
    
    # Compute specialty stats
    stats_query = """
        INSERT INTO clean.specialty_stats
        (medical_specialty, total_notes, avg_word_count, avg_char_count)
        SELECT 
            medical_specialty,
            COUNT(*) as total_notes,
            ROUND(AVG(word_count), 2) as avg_word_count,
            ROUND(AVG(char_count), 2) as avg_char_count
        FROM clean.clinical_notes
        GROUP BY medical_specialty
        ORDER BY total_notes DESC
    """
    
    cursor.execute(stats_query)
    conn.commit()
    
    # Show top specialties
    cursor.execute("""
        SELECT medical_specialty, total_notes, avg_word_count 
        FROM clean.specialty_stats 
        ORDER BY total_notes DESC 
        LIMIT 5
    """)
    
    print("\nTop 5 Specialties:")
    print(f"{'Specialty':<35} {'Notes':>8} {'Avg Words':>12}")
    print("-" * 57)
    for row in cursor.fetchall():
        print(f"{row[0]:<35} {row[1]:>8} {row[2]:>12}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    transform_notes()
