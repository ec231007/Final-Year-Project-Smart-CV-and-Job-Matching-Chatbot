import json
import os
from tqdm import tqdm
from eval_search import run_eval_search 

INPUT_FILE = "data/processed_eval_data.json"
OUTPUT_FILE = "data/raw_retrieval_results.json"

def run_mass_retrieval():
    # Load input data
    with open(INPUT_FILE, "r") as f:
        dataset = json.load(f)

    # CHECKPOINT LOGIC: Load existing work
    mass_results = []
    processed_files = set()
    
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r") as f:
                mass_results = json.load(f)
                processed_files = {node["filename"] for node in mass_results}
                print(f"⏩ Resuming from checkpoint. {len(processed_files)} CVs already done.")
        except Exception as e:
            print(f"⚠️ Checkpoint file unreadable, starting fresh: {e}")

    configs = [(False, False), (True, False), (False, True), (True, True)]

    for entry in tqdm(dataset, desc="🔍 Retrieval Progress"):
        fname = entry["filename"]
        if fname in processed_files:
            continue

        cv_eval_node = {
            "filename": fname,
            "ground_truth": entry["ground_truth_cat"],
            "eval_runs": []
        }

        try:
            for ner_on, llm_on in configs:
                run_data = run_eval_search(entry, NER_applied=ner_on, LLM_applied=llm_on)
                cv_eval_node["eval_runs"].append(run_data)
            
            mass_results.append(cv_eval_node)

            # Auto-save every 10 CVs
            if len(mass_results) % 10 == 0:
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(mass_results, f, indent=4)
                    
        except Exception as e:
            print(f"\n❌ Error on {fname}: {e}")
            with open(OUTPUT_FILE, "w") as f:
                json.dump(mass_results, f, indent=4)
            return # Exit and let you fix it

    # Final save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(mass_results, f, indent=4)
    print("\n✅ Mass retrieval complete!")

if __name__ == "__main__":
    run_mass_retrieval()
INPUT_FILE = "data/processed_eval_data.json"
OUTPUT_FILE = "data/raw_retrieval_results.json"

def run_mass_retrieval():
    # 1. Load Input Data
    with open(INPUT_FILE, "r") as f:
        dataset = json.load(f)

    # 2. Load Progress / Checkpoints
    mass_results = []
    done_filenames = set()
    
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            try:
                mass_results = json.load(f)
                done_filenames = {res["filename"] for res in mass_results}
                print(f"⏩ Resuming: {len(done_filenames)} CVs already processed.")
            except:
                print("⚠️ Output file corrupted or empty. Starting from scratch.")
                mass_results = []

    # All 4 possible system states
    configs = [
        (False, False), # Baseline
        (True, False),  # NER Only
        (False, True),  # LLM Boost Only
        (True, True)    # Full Hybrid
    ]

    # 3. Main Loop
    for entry in tqdm(dataset, desc="🔍 Running Mass Retrieval"):
        fname = entry["filename"]
        
        if fname in done_filenames:
            continue

        cv_eval_node = {
            "filename": fname,
            "ground_truth": entry["ground_truth_cat"],
            "eval_runs": []
        }

        try:
            for ner_toggle, llm_toggle in configs:
                run_data = run_eval_search(entry, NER_applied=ner_toggle, LLM_applied=llm_toggle)
                cv_eval_node["eval_runs"].append(run_data)
            
            mass_results.append(cv_eval_node)

            # 4. SAVE CHECKPOINT EVERY 5 CVs
            if len(mass_results) % 5 == 0:
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(mass_results, f, indent=4)
                    
        except Exception as e:
            print(f"\n❌ Fatal error on {fname}: {e}")
            # Save whatever we have before crashing
            with open(OUTPUT_FILE, "w") as f:
                json.dump(mass_results, f, indent=4)
            raise e

    # Final Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(mass_results, f, indent=4)
    
    print(f"\n✅ Done! Total processed: {len(mass_results)}")

if __name__ == "__main__":
    run_mass_retrieval()