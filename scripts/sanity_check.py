import json
from pathlib import Path

# 1. SETUP PATHS DYNAMICALLY
# This finds the folder where this script lives (scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent
# This goes one level up to the project root (smart-cv-chatbot/)
PROJECT_ROOT = SCRIPT_DIR.parent

# Define input/output paths relative to root
file_path =  PROJECT_ROOT / "data" / "processed" / "combined_cv_data.jsonl"

def sanity_check(limit=3):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                
                data = json.loads(line)
                text = data.get("text", "")
                annotations = data.get("annotations", [])
                
                print(f"\n{'='*30} RESUME #{i+1} {'='*30}")
                print(f"TOTAL TEXT LENGTH: {len(text)}")
                print(f"TEXT PREVIEW: {text[:100]}...")
                print(f"\nEXTRACTED ENTITIES:")
                print(f"{'START':<8} {'END':<8} {'LABEL':<15} {'EXTRACTED TEXT'}")
                print("-" * 60)
                
                for start, end, label in annotations:
                    # This is the "Truth Test"
                    extracted_word = text[start:end]
                    
                    # Highlighting if the extraction is empty or just whitespace
                    display_word = f"'{extracted_word}'" if extracted_word.strip() else "[EMPTY/SPACE]"
                    
                    print(f"{start:<8} {end:<8} {label:<15} {display_word}")
                
                print(f"{'='*70}\n")
                
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    sanity_check()