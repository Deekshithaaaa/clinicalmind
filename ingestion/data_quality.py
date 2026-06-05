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

def run_quality_checks():
    print("=== Running Data Quality Checks ===\n")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    checks = []
    
    # Check 1: Raw row count >= 4000
    cursor.execute("SELECT COUNT(*) FROM raw.clinical_notes")
    raw_count = cursor.fetchone()[0]
    checks.append((
        "Raw row count >= 4000",
        raw_count >= 4000,
        f"{raw_count} rows"
    ))
    
    # Check 2: Clean row count >= 4000
    cursor.execute("SELECT COUNT(*) FROM clean.clinical_notes")
    clean_count = cursor.fetchone()[0]
    checks.append((
        "Clean row count >= 4000",
        clean_count >= 4000,
        f"{clean_count} rows"
    ))
    
    # Check 3: No null transcriptions in clean layer
    cursor.execute("""
        SELECT COUNT(*) FROM clean.clinical_notes 
        WHERE transcription IS NULL
    """)
    null_transcriptions = cursor.fetchone()[0]
    checks.append((
        "No null transcriptions in clean layer",
        null_transcriptions == 0,
        f"{null_transcriptions} nulls"
    ))
    
    # Check 4: No null medical specialties
    cursor.execute("""
        SELECT COUNT(*) FROM clean.clinical_notes 
        WHERE medical_specialty IS NULL
    """)
    null_specialties = cursor.fetchone()[0]
    checks.append((
        "No null medical specialties",
        null_specialties == 0,
        f"{null_specialties} nulls"
    ))
    
    # Check 5: All notes have word count > 0
    cursor.execute("""
        SELECT COUNT(*) FROM clean.clinical_notes 
        WHERE word_count <= 0 OR word_count IS NULL
    """)
    invalid_wordcount = cursor.fetchone()[0]
    checks.append((
        "All notes have valid word count",
        invalid_wordcount == 0,
        f"{invalid_wordcount} invalid"
    ))
    
    # Check 6: No duplicate raw IDs in clean layer
    cursor.execute("""
        SELECT COUNT(*) FROM (
            SELECT raw_id, COUNT(*) 
            FROM clean.clinical_notes 
            GROUP BY raw_id 
            HAVING COUNT(*) > 1
        ) duplicates
    """)
    duplicates = cursor.fetchone()[0]
    checks.append((
        "No duplicate raw IDs in clean layer",
        duplicates == 0,
        f"{duplicates} duplicates"
    ))
    
    # Check 7: Specialty stats table has data
    cursor.execute("SELECT COUNT(*) FROM clean.specialty_stats")
    stats_count = cursor.fetchone()[0]
    checks.append((
        "Specialty stats table has data",
        stats_count > 0,
        f"{stats_count} specialties"
    ))
    
    # Check 8: Average word count is reasonable (> 50 words)
    cursor.execute("""
        SELECT AVG(word_count) FROM clean.clinical_notes
    """)
    avg_words = cursor.fetchone()[0]
    checks.append((
        "Average word count > 50",
        avg_words > 50,
        f"{avg_words:.1f} avg words"
    ))
    
    # Print results
    passed = 0
    failed = 0
    
    for check_name, result, detail in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {check_name:<45} | {detail}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{len(checks)} checks passed")
    
    if failed == 0:
        print("🎉 All quality checks passed!")
    else:
        print(f"⚠️  {failed} check(s) failed — review before proceeding")
    
    cursor.close()
    conn.close()
    
    return failed == 0

if __name__ == "__main__":
    run_quality_checks()
