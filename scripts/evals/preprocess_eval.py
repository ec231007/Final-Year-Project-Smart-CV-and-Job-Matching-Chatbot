import os
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from resume_ner_bert import parse_resume_ner_bert
from resume_parser_util import extract_text_from_file
from groq_prompter import get_filter_json, get_search_query_llm

# --- SETTINGS ---
# Set this to True once to clear the old format data, then False to resume
CLEAR_ON_START = False

# 1. SETUP PATHS
SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "data" / "Unprocessed_cv" / "data"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed_eval_data.json"

def preprocess_with_retry():
    # 1. CLEAR OR LOAD PROGRESS
    if CLEAR_ON_START and os.path.exists(OUTPUT_FILE):
        print("--- Clearing existing dataset for a fresh start ---")
        os.remove(OUTPUT_FILE)
        processed_data = []
    elif os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            try:
                processed_data = json.load(f)
            except:
                processed_data = []
    else:
        processed_data = []

    done_files = {item['filename'] for item in processed_data}
    categories = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

    for cat in categories:
        cat_path = os.path.join(DATASET_DIR, cat)
        # Process first 50 per category for evaluation
        files = os.listdir(cat_path)[:50] 
        
        print(f"\nProcessing category: {cat}")
        for f_name in tqdm(files):
            if f_name in done_files:
                continue 

            file_path = os.path.join(cat_path, f_name)
            
            # --- API RETRY LOGIC ---
            success = False
            retry_count = 0
            
            while not success:
                try:
                    text = extract_text_from_file(file_path)
                    
                    # Run NER (Local BERT - No API tokens used)
                    ner_output = parse_resume_ner_bert(text)
                    
                    # Run Intent Extraction (Returns Tuple: data, model)
                    intent_data, intent_model = get_filter_json(
                        f"RESUME: {text[:1000]}\nUSER PREFERENCES: Find a job matching my profile"
                    )
                    
                    # Run LLM Boost (Returns Tuple: query, model)
                    llm_boost_query, boost_model = get_search_query_llm(text, "")

                    new_entry = {
                        "filename": f_name,
                        "ground_truth_cat": cat,
                        "resume_text": text[:2000], # Store trimmed text for eval reference
                        "ner_tags": [tag for sublist in ner_output.values() for tag in sublist],
                        "intent": intent_data,
                        "intent_model_used": intent_model,
                        "llm_boost": llm_boost_query,
                        "boost_model_used": boost_model,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }

                    processed_data.append(new_entry)
                    done_files.add(f_name)
                    
                    # SAVE IMMEDIATELY (Checkpoint)
                    with open(OUTPUT_FILE, "w") as out:
                        json.dump(processed_data, out, indent=4)
                    
                    success = True 

                except Exception as e:
                    if "429" in str(e):
                        retry_count += 1
                        if retry_count > 10:
                            print(f"\n🛑 [FATAL] Rate limit hit 10 times consecutively. Assuming Daily Cap.")
                            print("Saving progress and exiting. Run again tomorrow!")
                            sys.exit(0) 
                        
                        print(f"\n⚠️ Rate limit hit (Attempt {retry_count}/10). Napping for 60s...")
                        time.sleep(60)
                    else:
                        print(f"\n❌ Permanent error on {f_name}: {e}")
                        break # Move to next file if it's not a rate limit issue

if __name__ == "__main__":
    preprocess_with_retry()