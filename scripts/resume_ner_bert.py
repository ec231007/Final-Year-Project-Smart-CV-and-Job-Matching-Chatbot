"""
Resume NER using a BERT model trained for resumes (yashpwr/resume-ner-bert-v2).

This model outputs resume-specific labels: Skills, Designation, Location, Degree,
College Name, etc., and avoids the en_core_web_sm mistakes (e.g. Python as GPE).

Requires: pip install transformers torch
Output: {"roles": [], "skills": [], "education": [], "locations": []}
"""

from typing import Dict, List, Optional, Set

# Lazy-load to avoid importing torch/transformers until needed
_ner_pipeline = None

RESUME_NER_MODEL = "yashpwr/resume-ner-bert-v2"
# Model is trained with max_length 128; we chunk to stay under that
CHUNK_SIZE = 120
CHUNK_OVERLAP = 30

# Map BERT model labels to our output keys (model uses these exact or similar)
LABEL_TO_KEY = {
    "Designation": "roles",
    "Skills": "skills",
    "Degree": "education",
    "College Name": "education",
    "Location": "locations",
    # Ignored for search: Name, Years of Experience, Companies worked at, Email, Phone, Graduation Year
}


def _label_to_key(label: str) -> Optional[str]:
    if not label:
        return None
    key = LABEL_TO_KEY.get(label)
    if key:
        return key
    # Try normalized (e.g. B-Degree -> Degree)
    normalized = label.replace("B-", "").replace("I-", "").strip()
    return LABEL_TO_KEY.get(normalized)


def _get_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        try:
            from transformers import pipeline
            _ner_pipeline = pipeline(
                "token-classification",
                model=RESUME_NER_MODEL,
                aggregation_strategy="simple",
                device=-1,  # CPU
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load resume NER model {RESUME_NER_MODEL}. "
                "Install with: pip install transformers torch"
            ) from e
    return _ner_pipeline


def _normalize(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    return " ".join(s.split()).strip()


# Skip entities with score below this
CONFIDENCE_THRESHOLD = 0.5
# Section headers / junk the model sometimes tags as Skills
SKILLS_BLOCKLIST = {"skills", "education", "learning education", ",", ""}
# 2-letter state codes often tagged as Degree; treat as location noise
STATE_CODES = {"ma", "ca", "ny", "tx", "fl", "il", "pa", "oh", "ga", "nc", "mi", "nj", "va", "wa", "az", "co", "or", "tn", "mo", "md"}


def _unique(lst: List[str], min_len: int = 1) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in lst:
        x = _normalize(x)
        if len(x) < min_len:
            continue
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def _chunk_text(text: str, tokenizer, max_tokens: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks by token count."""
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text)
        start = end - overlap
        if start >= len(tokens):
            break
    return chunks if chunks else [text]


def parse_resume_ner_bert(
    resume_text: str,
    *,
    max_roles: int = 20,
    max_skills: int = 50,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Dict[str, List[str]]:
    """
    Extract roles, skills, education, locations using resume-specific BERT NER.

    Uses yashpwr/resume-ner-bert-v2 (25 entity types, ~90% F1 on resumes).
    Long text is split into overlapping chunks; entities are merged and deduped.
    """
    text = _normalize(resume_text)
    if not text:
        return {"roles": [], "skills": [], "education": [], "locations": []}

    pipe = _get_pipeline()
    tokenizer = pipe.tokenizer

    # Chunk if needed (model trained with max_length 128)
    chunks = _chunk_text(text, tokenizer, chunk_size, chunk_overlap)

    collected: Dict[str, List[str]] = {
        "roles": [],
        "skills": [],
        "education": [],
        "locations": [],
    }

    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            # pipeline returns list of {entity_group, word, start, end, score}
            out = pipe(chunk)
        except Exception:
            continue
        if not out:
            continue
        for item in out:
            score = item.get("score", 0.0)
            if score < CONFIDENCE_THRESHOLD:
                continue
            label = item.get("entity_group") or item.get("entity") or ""
            word = _normalize(item.get("word", "")).strip(".,; ")
            key = _label_to_key(label)
            if not word or not key:
                continue
            if key == "locations" and word.lower() in {"python", "java", "sql", "aws", "ca"}:
                continue
            if key == "education" and word.lower() in STATE_CODES:
                continue
            if key == "skills":
                if word.lower() in SKILLS_BLOCKLIST or len(word) < 2:
                    continue
                # Split comma-separated skill lists into individual skills
                for part in word.split(","):
                    part = _normalize(part).strip(".,; ")
                    if len(part) >= 2 and part.lower() not in SKILLS_BLOCKLIST:
                        collected["skills"].append(part)
                continue
            collected[key].append(word)

    return {
        "roles": _unique(collected["roles"], min_len=2)[:max_roles],
        "skills": _unique(collected["skills"], min_len=2)[:max_skills],
        "education": _unique(collected["education"], min_len=2),
        "locations": _unique(collected["locations"], min_len=2),
    }


def parse_resume_file_bert(file_path: str, **kwargs) -> Dict[str, List[str]]:
    """Load resume from file (PDF/DOCX), extract text, run BERT NER."""
    from resume_parser_util import extract_text_from_file
    text = extract_text_from_file(file_path)
    return parse_resume_ner_bert(text, **kwargs)


if __name__ == "__main__":
    SAMPLE = """
    John Doe
    Summary
    Software engineer with 5 years of experience in Python and cloud systems.
    Experience
    Senior Software Engineer
    Jan 2020 to Current
    Tech Company Inc. — New York, NY
    Built APIs and data pipelines. Led a team of 4.
    Software Developer
    Mar 2018 to Dec 2019
    Startup Co — San Francisco, CA
    Skills
    Python, Java, AWS, SQL, REST APIs, Docker, Kubernetes, machine learning
    Education
    Bachelor of Science in Computer Science 2016
    State University — Boston, MA
    """
    print("Running BERT resume NER (first run may download the model)...")
    result = parse_resume_ner_bert(SAMPLE)
    print("\n--- Extracted tags ---")
    for k, v in result.items():
        print(f"{k}: {v}")
