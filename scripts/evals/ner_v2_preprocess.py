import json
import os
from pathlib import Path
import sys
import time
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from resume_ner_bert_v2 import parse_resume_ner_bert
from resume_parser_util import extract_text_from_file

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_JSON = PROJECT_ROOT / "data" / "processed_eval_data.json"
OUTPUT_JSON = PROJECT_ROOT / "data" / "processed_eval_data_v2.json"
RESUME_DIR = PROJECT_ROOT / "data" / "Unprocessed_cv" / "data"

def enrich_dataset():
    if not os.path.exists(INPUT_JSON):
        print(f"❌ Error: {INPUT_JSON} not found.")
        return

    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    total_records = len(data)
    print(f"🚀 Starting Enrichment of {total_records} records...")
    print(f"📂 Looking for resumes in: {RESUME_DIR}\n")

    start_time = time.time()
    success_count = 0
    missing_count = 0
    
    # tqdm creates a progress bar with: % complete | items/sec | ETA
    for i, entry in enumerate(tqdm(data, desc="Enriching NER v2", unit="cv")):
        filename = entry.get("filename")
        category = entry.get("ground_truth_cat") 
        file_path = os.path.join(RESUME_DIR, category, filename)

        if os.path.exists(file_path):
            try:
                # 1. Extract and Run NER
                text = extract_text_from_file(file_path)
                v2_results = parse_resume_ner_bert(text)
                
                # 2. Flatten tags
                v2_flat_tags = [tag for sublist in v2_results.values() for tag in sublist]
                entry["ner_tags_v2"] = v2_flat_tags
                success_count += 1
                
            except Exception as e:
                # Use tqdm.write to prevent breaking the progress bar line
                tqdm.write(f"⚠️ Error on {filename}: {e}")
                entry["ner_tags_v2"] = []
        else:
            missing_count += 1
            entry["ner_tags_v2"] = []

        # --- CHECKPOINT SAVE ---
        # Saves every 25 records so you don't lose progress
        if (i + 1) % 25 == 0:
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(data, f, indent=4)

    # --- FINAL SAVE ---
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=4)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*40)
    print(f"✅ ENRICHMENT COMPLETE")
    print(f"⏱️ Total Time: {elapsed/60:.2f} minutes")
    print(f"📊 Successfully Processed: {success_count}")
    print(f"❓ Files Not Found: {missing_count}")
    print(f"💾 File Saved: {OUTPUT_JSON}")
    
    # Quick insight into the results
    avg_v1 = sum(len(e.get('ner_tags', [])) for e in data) / total_records
    avg_v2 = sum(len(e.get('ner_tags_v2', [])) for e in data) / total_records
    print(f"📈 Avg Tags Per CV (Old Model): {avg_v1:.1f}")
    print(f"📈 Avg Tags Per CV (New Model): {avg_v2:.1f}")
    print("="*40)

if __name__ == "__main__":
    enrich_dataset()