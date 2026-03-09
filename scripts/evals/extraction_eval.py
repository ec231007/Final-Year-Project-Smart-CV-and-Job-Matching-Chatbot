import json
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# the same embedding model used in DB for consistency
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_intent_fill_rate(intent):
    """Calculates what % of the 4 key fields were successfully extracted."""
    if not intent: return 0.0
    # Fields: experience, work_type, location, title
    fields = [intent.get("experience"), intent.get("work_type"), 
              intent.get("location"), intent.get("title")]
    filled = sum(1 for f in fields if f and f != [])
    return filled / 4.0

def calculate_extraction_metrics(json_input_path, csv_output_path):
    with open(json_input_path, 'r') as f:
        data = json.load(f)

    rows = []
    print(f"Reading {len(data)} records...")

    # Pre-calculate category embeddings for speed
    unique_cats = list(set(d['ground_truth_cat'] for d in data))
    cat_embs = {cat: model.encode(cat.replace("_", " "), convert_to_tensor=True) for cat in unique_cats}

    for entry in data:
        # 1. Basic Stats
        filename = entry.get("filename")
        category = entry.get("ground_truth_cat")
        ner_count = len(entry.get("ner_tags", []))
        fill_rate = get_intent_fill_rate(entry.get("intent"))
        
        # 2. Semantic Similarity (Ground Truth vs Extracted Title)
        extracted_title = entry.get("intent", {}).get("title") or ""
        similarity = 0.0
        if extracted_title:
            title_emb = model.encode(extracted_title, convert_to_tensor=True)
            similarity = util.cos_sim(cat_embs[category], title_emb).flatten()[0].item()

        # 3. Build Row
        rows.append({
            "filename": filename,
            "category": category,
            "ner_tag_count": ner_count,
            "intent_fill_rate": fill_rate,
            "semantic_similarity": round(similarity, 4),
            "intent_model": entry.get("intent_model_used"),
            "boost_model": entry.get("boost_model_used")
        })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save Master CSV
    df.to_csv(csv_output_path, index=False)
    print(f"✅ Metrics saved to {csv_output_path}")

    # Display Category Summary for a quick check
    summary = df.groupby('category').agg({
        'ner_tag_count': 'mean',
        'intent_fill_rate': 'mean',
        'semantic_similarity': 'mean'
    }).round(3)
    
    print("\n--- Category Summary (Averages) ---")
    print(summary)

if __name__ == "__main__":
    INPUT_JSON = "data/processed_eval_data.json"
    OUTPUT_CSV = "data/extraction_metrics_report.csv"
    calculate_extraction_metrics(INPUT_JSON, OUTPUT_CSV)