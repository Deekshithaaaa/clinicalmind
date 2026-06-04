import requests
import os

def download_mtsamples():
    print("Downloading MTSamples dataset...")
    
    url = "https://raw.githubusercontent.com/socd06/medical-nlp/master/data/mtsamples.csv"
    
    os.makedirs("data/raw", exist_ok=True)
    
    response = requests.get(url)
    
    if response.status_code == 200:
        with open("data/raw/mtsamples.csv", "wb") as f:
            f.write(response.content)
        print("✅ MTSamples downloaded successfully!")
        print(f"   Saved to: data/raw/mtsamples.csv")
    else:
        print(f"❌ Download failed: {response.status_code}")

if __name__ == "__main__":
    download_mtsamples()
