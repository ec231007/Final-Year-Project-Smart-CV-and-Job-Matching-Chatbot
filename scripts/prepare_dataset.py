import os
import json
import pandas as pd
import spacy
from pathlib import Path
from tqdm import tqdm

# 1. SETUP PATHS DYNAMICALLY
# This finds the folder where this script lives (scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent
# This goes one level up to the project root (smart-cv-chatbot/)
PROJECT_ROOT = SCRIPT_DIR.parent

# Define input/output paths relative to root
CSV_PATH = PROJECT_ROOT / "data" / "Unprocessed_cv" / "Resume" / "Resume.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "combined_cv_data.jsonl"

nlp = spacy.load("en_core_web_sm")

class ResumeDataConverter:
    def __init__(self, output_file):
        self.output_file = output_file
        # Map spaCy's default labels to your project's 14 categories
        self.label_map = {"PERSON": "PERSON", "GPE": "LOCATION", "ORG": "COMPANY"}

    def clean_text(self, text):
        if not text: return ""
        # Keep it simple for NER: normalize whitespace but keep casing
        return " ".join(text.split())

    def get_bootstrap_annotations(self, text):
        doc = nlp(text)
        return [[ent.start_char, ent.end_char, self.label_map.get(ent.label_, "OTHER")] 
                for ent in doc.ents]

    def process_csv(self, path):
        if not path.exists():
            print(f"ERROR: Could not find CSV at {path}")
            return []

        df = pd.read_csv(path)
        processed_data = []

        print(f"Processing {len(df)} records from CSV...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            clean_txt = self.clean_text(str(row['Resume_str']))
            
            # Here is the metadata you suggested!
            entry = {
                "text": clean_txt,
                "annotations": self.get_bootstrap_annotations(clean_txt),
                "metadata": {
                    "source_id": str(row['ID']),
                    "category": row['Category'],  # For LLM justification
                    "label": "unprocessed_source"
                }
            }
            processed_data.append(entry)
        return processed_data

    def save(self, data):
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Success! Saved to {self.output_file}")

if __name__ == "__main__":
    converter = ResumeDataConverter(OUTPUT_PATH)
    data = converter.process_csv(CSV_PATH)
    if data:
        converter.save(data)