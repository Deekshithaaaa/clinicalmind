from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'clinicalmind',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_ingestion():
    import pandas as pd
    import psycopg2
    from psycopg2.extras import execute_batch

    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="clinicalmind",
        user="clinicalmind",
        password="clinicalmind123"
    )
    cursor = conn.cursor()

    df = pd.read_csv("/opt/airflow/data/raw/mtsamples.csv")
    df = df.dropna(subset=['transcription'])

    cursor.execute("TRUNCATE TABLE raw.clinical_notes RESTART IDENTITY CASCADE")
    conn.commit()

    records = []
    for _, row in df.iterrows():
        records.append((
            str(row.get('description', '')).strip(),
            str(row.get('medical_specialty', '')).strip(),
            str(row.get('sample_name', '')).strip(),
            str(row.get('transcription', '')).strip(),
            str(row.get('keywords', '')).strip()
        ))

    insert_query = """
        INSERT INTO raw.clinical_notes
        (description, medical_specialty, sample_name, transcription, keywords)
        VALUES (%s, %s, %s, %s, %s)
    """
    execute_batch(cursor, insert_query, records, page_size=100)
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM raw.clinical_notes")
    count = cursor.fetchone()[0]
    print(f"✅ Ingested {count} records")

    cursor.close()
    conn.close()

def run_transformation():
    import psycopg2

    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="clinicalmind",
        user="clinicalmind",
        password="clinicalmind123"
    )
    cursor = conn.cursor()

    cursor.execute("TRUNCATE TABLE clean.clinical_notes RESTART IDENTITY CASCADE")
    cursor.execute("TRUNCATE TABLE clean.specialty_stats RESTART IDENTITY CASCADE")
    conn.commit()

    cursor.execute("""
        INSERT INTO clean.clinical_notes
        (raw_id, medical_specialty, sample_name, transcription,
         word_count, char_count, specialty_category)
        SELECT
            id,
            TRIM(medical_specialty),
            TRIM(sample_name),
            TRIM(transcription),
            array_length(string_to_array(TRIM(transcription), ' '), 1),
            LENGTH(TRIM(transcription)),
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
            END
        FROM raw.clinical_notes
        WHERE transcription IS NOT NULL
        AND LENGTH(TRIM(transcription)) > 50
    """)

    cursor.execute("""
        INSERT INTO clean.specialty_stats
        (medical_specialty, total_notes, avg_word_count, avg_char_count)
        SELECT
            medical_specialty,
            COUNT(*),
            ROUND(AVG(word_count), 2),
            ROUND(AVG(char_count), 2)
        FROM clean.clinical_notes
        GROUP BY medical_specialty
    """)
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM clean.clinical_notes")
    count = cursor.fetchone()[0]
    print(f"✅ Transformed {count} records")

    cursor.close()
    conn.close()

def run_quality_checks():
    import psycopg2

    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="clinicalmind",
        user="clinicalmind",
        password="clinicalmind123"
    )
    cursor = conn.cursor()

    checks = []

    cursor.execute("SELECT COUNT(*) FROM raw.clinical_notes")
    checks.append(("Raw count >= 4000", cursor.fetchone()[0] >= 4000))

    cursor.execute("SELECT COUNT(*) FROM clean.clinical_notes")
    checks.append(("Clean count >= 4000", cursor.fetchone()[0] >= 4000))

    cursor.execute("SELECT COUNT(*) FROM clean.clinical_notes WHERE transcription IS NULL")
    checks.append(("No null transcriptions", cursor.fetchone()[0] == 0))

    cursor.execute("SELECT COUNT(*) FROM clean.clinical_notes WHERE medical_specialty IS NULL")
    checks.append(("No null specialties", cursor.fetchone()[0] == 0))

    cursor.execute("SELECT COUNT(*) FROM clean.clinical_notes WHERE word_count <= 0")
    checks.append(("Valid word counts", cursor.fetchone()[0] == 0))

    cursor.execute("SELECT COUNT(*) FROM clean.specialty_stats")
    checks.append(("Specialty stats populated", cursor.fetchone()[0] > 0))

    failed = [c for c in checks if not c[1]]
    if failed:
        raise ValueError(f"Quality checks failed: {[c[0] for c in failed]}")

    print(f"✅ All {len(checks)} quality checks passed!")
    cursor.close()
    conn.close()

with DAG(
    'clinical_pipeline',
    default_args=default_args,
    description='ClinicalMind ETL Pipeline',
    schedule_interval='@daily',
    catchup=False,
    tags=['clinicalmind', 'healthcare']
) as dag:

    ingest = PythonOperator(
        task_id='ingest_clinical_notes',
        python_callable=run_ingestion,
    )

    transform = PythonOperator(
        task_id='transform_clinical_notes',
        python_callable=run_transformation,
    )

    quality = PythonOperator(
        task_id='run_quality_checks',
        python_callable=run_quality_checks,
    )

    ingest >> transform >> quality
