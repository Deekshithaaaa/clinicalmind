import pandas as pd

def explore():
    df = pd.read_csv("data/raw/mtsamples.csv")
    
    print("=== MTSamples Dataset Overview ===")
    print(f"Total records: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nMedical specialties: {df['medical_specialty'].nunique()}")
    print(f"\nTop 10 specialties:")
    print(df['medical_specialty'].value_counts().head(10))
    print(f"\nSample clinical note:")
    print("-" * 50)
    print(df['transcription'].iloc[0][:500])
    print("-" * 50)
    print(f"\nMissing values:")
    print(df.isnull().sum())

if __name__ == "__main__":
    explore()
