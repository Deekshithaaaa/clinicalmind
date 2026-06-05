-- Raw layer: stores data exactly as ingested
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS clean;

-- Raw clinical notes table
CREATE TABLE IF NOT EXISTS raw.clinical_notes (
    id SERIAL PRIMARY KEY,
    description TEXT,
    medical_specialty VARCHAR(100),
    sample_name TEXT,
    transcription TEXT,
    keywords TEXT,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Clean layer: transformed and validated notes
CREATE TABLE IF NOT EXISTS clean.clinical_notes (
    id SERIAL PRIMARY KEY,
    raw_id INTEGER REFERENCES raw.clinical_notes(id),
    medical_specialty VARCHAR(100),
    sample_name TEXT,
    transcription TEXT,
    word_count INTEGER,
    char_count INTEGER,
    specialty_category VARCHAR(50),
    cleaned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analytics layer: aggregated insights
CREATE TABLE IF NOT EXISTS clean.specialty_stats (
    id SERIAL PRIMARY KEY,
    medical_specialty VARCHAR(100),
    total_notes INTEGER,
    avg_word_count NUMERIC(10,2),
    avg_char_count NUMERIC(10,2),
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
