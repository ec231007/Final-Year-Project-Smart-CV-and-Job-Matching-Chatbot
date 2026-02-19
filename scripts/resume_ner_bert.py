"""
Resume NER using a BERT model trained for resumes (yashpwr/resume-ner-bert-v2).

This model outputs resume-specific labels: Skills, Designation, Location, Degree,
College Name, etc., and avoids the en_core_web_sm mistakes (e.g. Python as GPE).

Requires: pip install transformers torch
Output: {"roles": [], "skills": [], "education": [], "locations": []}
"""

import re
from typing import Dict, List
from transformers import pipeline
from resume_parser_util import extract_text_from_file

# Lazy-load to avoid importing torch/transformers until needed
_ner_pipeline = None

RESUME_NER_MODEL = "yashpwr/resume-ner-bert-v2"

def _get_pipeline():
    # 'simple' is often too basic; 'first' or 'max' handles subwords better for this model
    return pipeline(
        "token-classification",
        model=RESUME_NER_MODEL,
        aggregation_strategy="first", 
        device=-1
    )

def _clean_text(text: str) -> str:
    """Fixes BERT subword fragments and cleans common junk."""
    # Fix broken subwords that simple aggregation missed
    text = text.replace(" ##", "").replace("##", "")
    # Remove weird artifacts like "( 100 % )"
    text = re.sub(r'\(.*?\)', '', text)
    # Strip non-alphanumeric at ends but keep internal spaces
    return text.strip(".,;•· ")

def parse_resume_ner_bert(resume_text: str) -> Dict[str, List[str]]:
    pipe = _get_pipeline()
    
    # We chunk manually because the model has a hard 512 token limit
    # We split by sentences or lines to keep context
    lines = resume_text.split('\n')
    chunks = []
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) < 500: # Stay safe under 512
            current_chunk += line + " "
        else:
            chunks.append(current_chunk)
            current_chunk = line + " "
    chunks.append(current_chunk)

    collected = {"roles": [], "skills": [], "education": [], "locations": []}
    
    # Map labels to our keys
    MAP = {
        "Designation": "roles", "Skills": "skills", 
        "Degree": "education", "College Name": "education", "Location": "locations"
    }

    for chunk in chunks:
        if not chunk.strip(): continue
        entities = pipe(chunk)
        
        for ent in entities:
            key = MAP.get(ent['entity_group'])
            word = _clean_text(ent['word'])
            
            if not key or len(word) < 2: continue

            # Quality Control: Filter out full sentences misclassified as skills
            if key == "skills":
                if len(word.split()) > 3: continue # Skills are usually 1-3 words
                if any(char in word for char in ['.', '!', '?']): continue
            
            collected[key].append(word)

    # Dedup and Clean
    return {k: sorted(list(set(v))) for k, v in collected.items()}


def parse_resume_file_bert(file_path: str, **kwargs) -> Dict[str, List[str]]:
    """Load resume from file (PDF/DOCX), extract text, run BERT NER."""
    from resume_parser_util import extract_text_from_file
    text = extract_text_from_file(file_path)
    return parse_resume_ner_bert(text, **kwargs)


if __name__ == "__main__":
    SAMPLE =r"C:\Vasanth\Important stuff\Resumes\Vasanth Subramanian Resume.pdf"
    print("Running BERT resume NER (first run may download the model)...")
    SAMPLE_text = extract_text_from_file(SAMPLE)
    result = parse_resume_ner_bert(SAMPLE_text)
    print("\n--- Extracted tags ---")
    for k, v in result.items():
        print(f"{k}: {v}")
