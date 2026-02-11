import pandas as pd
import json
from pathlib import Path

# 1. SETUP PATHS
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSV_PATH = PROJECT_ROOT / "data" / "LinkedIn" / "postings.csv"
CACHE_OUTPUT = SCRIPT_DIR / "metadata_cache.json"

def generate_metadata_cache():
    print(f"🔄 Loading CSV from: {CSV_PATH}")
    
    # Only load necessary columns to keep memory usage low
    cols = ['location', 'formatted_experience_level', 'work_type']
    df = pd.read_csv(CSV_PATH, usecols=cols).fillna("UNKNOWN")

    # 2. EXTRACT UNIQUE VALUES
    # We use a dictionary to keep it organized
    cache = {
        "locations": sorted(df['location'].unique().tolist()),
        "experience_levels": sorted(df['formatted_experience_level'].unique().tolist()),
        "work_types": sorted(df['work_type'].unique().tolist())
    }

    # 3. SAVE TO JSON
    with open(CACHE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=4)
    
    print(f"Cache generated successfully at: {CACHE_OUTPUT}")
    print(f"Found {len(cache['locations'])} unique locations.")

if __name__ == "__main__":
    generate_metadata_cache()