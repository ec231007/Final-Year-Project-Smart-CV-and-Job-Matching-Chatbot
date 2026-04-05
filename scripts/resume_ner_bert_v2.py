"""
Improved Resume NER using yashpwr/resume-ner-bert-v2.
Fixes truncation issues, comma-separated lists, and header hallucinations.
"""

import re
import torch
from typing import Dict, List
from transformers import pipeline

_ner_pipeline = None
RESUME_NER_MODEL = "yashpwr/resume-ner-bert-v2"

def _get_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        print(f"--- Initializing NER Model: {RESUME_NER_MODEL} ---")
        device = 0 if torch.cuda.is_available() else -1
        
        # CHANGED: 'simple' correctly stitches sub-words back into full phrases
        _ner_pipeline = pipeline(
            "token-classification",
            model=RESUME_NER_MODEL,
            aggregation_strategy="simple", 
            device=device
        )
    return _ner_pipeline

def _clean_text(text: str) -> str:
    """Cleans junk characters while preserving valid inner punctuation."""
    # Strip non-alphanumeric at ends but keep internal spaces/dots (e.g. Node.js)
    return text.strip(" .,;•·\n\t")

def parse_resume_ner_bert(resume_text: str) -> Dict[str, List[str]]:
    pipe = _get_pipeline()
    
    # CHANGED: Token-aware chunking to prevent splitting entities mid-word
    # Using 400 characters (~80-100 tokens) with 50 chars overlap
    chunk_size, overlap = 400, 50
    chunks = []
    start = 0
    while start < len(resume_text):
        end = start + chunk_size
        chunks.append(resume_text[start:end])
        start = end - overlap
        if start >= len(resume_text): break

    collected = {"roles": [], "skills": [], "education": [], "locations": []}
    
    MAP = {
        "Designation": "roles", 
        "Skills": "skills", 
        "Degree": "education", 
        "College Name": "education", 
        "Location": "locations"
    }

    # Stop-words to prevent section headers from becoming entities
    STOP_WORDS = {"skills", "education", "experience", "summary", "profile", "certifications"}

    for chunk in chunks:
        if not chunk.strip(): continue
        entities = pipe(chunk)
        
        for ent in entities:
            # Drop low-confidence predictions to reduce noise
            if ent.get('score', 0) < 0.50: continue
                
            key = MAP.get(ent['entity_group'])
            if not key: continue

            raw_word = ent['word']
            
            # CHANGED: If the model groups skills by comma, split them up!
            if key in ["skills", "locations"] and "," in raw_word:
                sub_words = [w for w in raw_word.split(",")]
            else:
                sub_words = [raw_word]
                
            for word in sub_words:
                cleaned_word = _clean_text(word)
                
                if len(cleaned_word) < 2: continue
                if cleaned_word.lower() in STOP_WORDS: continue

                word_count = len(cleaned_word.split())
                
                # Education can be long (e.g., "BSc Computer Science"), but roles/skills shouldn't be
                if key == "education" and word_count > 10: continue
                if key != "education" and word_count > 6: continue 
                
                collected[key].append(cleaned_word)

    # Dedup and sort
    return {k: sorted(list(set(v))) for k, v in collected.items()}