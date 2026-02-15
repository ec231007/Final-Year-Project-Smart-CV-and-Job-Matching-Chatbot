"""
Resume NER Parser — Extract structured tags from resume text using spaCy + section rules.

Uses en_core_web_sm for entities (GPE, LOC, ORG, DATE) and section-aware rules for:
- roles: job titles (Experience section)
- skills: comma/newline-separated items (Skills section) + noun chunks in Skills
- education: institutions and degree-like phrases (Education section)
- locations: NER GPE/LOC with a blocklist to filter mis-tags (e.g. Python as GPE)

Output: {"roles": [], "skills": [], "education": [], "locations": []}

Note: This spaCy-based parser was later replaced by the BERT resume NER model
(resume_ner_bert.py) which gives better accuracy on resume-specific entities.
"""

import re
import spacy
from typing import Dict, List, Optional, Set

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

# Section headers (case-insensitive). Longest first for matching.
SECTION_HEADERS = sorted([
    "experience", "work experience", "employment", "professional experience",
    "education", "academic", "qualifications", "training",
    "skills", "technical skills", "core competencies", "summary", "objective",
    "accomplishments", "projects", "certifications",
], key=len, reverse=True)


def _normalize(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    return " ".join(s.split()).strip()


def _unique_lower(lst: List[str], min_len: int = 2) -> List[str]:
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


def _find_sections(text: str) -> Dict[str, str]:
    """Split resume into sections by common headers. Do not normalize full text (keeps newlines)."""
    if not text or not text.strip():
        return {}
    lines = text.split("\n")
    sections: Dict[str, str] = {}
    current_header: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_header:
                current_lines.append("")
            continue
        lower = stripped.lower()
        matched_header = None
        for h in SECTION_HEADERS:
            if lower == h or lower.startswith(h + ":") or lower.startswith(h + " "):
                matched_header = h
                break
        if matched_header:
            if current_header:
                sections[current_header] = "\n".join(current_lines)
            current_header = matched_header
            rest = stripped[len(matched_header):].lstrip(": ").strip()
            current_lines = [rest] if rest else []
        else:
            if current_header:
                current_lines.append(stripped)
    if current_header:
        sections[current_header] = "\n".join(current_lines)
    return sections


def _extract_roles_from_experience(experience_text: str) -> List[str]:
    roles: List[str] = []
    company_indicators = ("company name", "inc.", " inc", " llc", " ltd", "co.", " co ", " co", " — ", "city", "state")
    date_only = re.compile(r"^(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{4}|\d{4})\s*$", re.I)
    lines = [l.strip() for l in experience_text.split("\n") if l.strip()]
    for line in lines:
        if re.match(r"^(?:\d{1,2}[/-])?\d{1,2}[/-]\d{2,4}\s+(?:to|–|-)\s+", line, re.I):
            continue
        if date_only.match(line):
            continue
        if "company name" in line.lower() or any(c in line.lower() for c in company_indicators):
            continue
        for sep in [" at ", " to ", " \u2013 ", " - ", " \u2014 ", " — "]:
            if sep in line:
                line = line.split(sep)[0].strip()
                break
        if len(line) > 45 or line.endswith("."):
            continue
        if any(line.lower().startswith(w) for w in ("built", "led", "managed", "developed", "designed", "created")):
            continue
        if 2 <= len(line) <= 60 and not line.replace(" ", "").isdigit():
            roles.append(line)
    return _unique_lower(roles, min_len=2)


def _extract_skills_from_skills_section(skills_text: str) -> List[str]:
    if not skills_text:
        return []
    raw = re.split(r"[,;\n\u2022\u2023\u25e6\*]\s*", skills_text)
    tokens = [_normalize(t) for t in raw if _normalize(t)]
    return _unique_lower(tokens, min_len=2)


def _extract_education_phrases(education_text: str) -> List[str]:
    if not education_text:
        return []
    phrases: List[str] = []
    degree_pattern = re.compile(
        r"\b(?:Bachelor|B\.?S\.?|B\.?A\.?|Master|M\.?S\.?|M\.?A\.?|MBA|PhD|Ph\.?D\.?|"
        r"Associate|Certificate|Diploma|High School)\b[^.\n]*",
        re.I,
    )
    for m in degree_pattern.finditer(education_text):
        phrases.append(_normalize(m.group(0)))
    return _unique_lower(phrases, min_len=3)


def parse_resume_ner(
    resume_text: str,
    *,
    use_noun_chunks_for_skills: bool = True,
    max_skills: int = 50,
    max_roles: int = 20,
) -> Dict[str, List[str]]:
    """Parse resume text into roles, skills, education, locations using spaCy + section rules."""
    if nlp is None:
        raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    text = _normalize(resume_text)
    if not text:
        return {"roles": [], "skills": [], "education": [], "locations": []}

    sections = _find_sections(resume_text)
    experience_text = sections.get("experience", "") or sections.get("work experience", "") or sections.get("employment", "")
    skills_section_text = sections.get("skills", "") or sections.get("technical skills", "") or sections.get("core competencies", "")
    education_text = sections.get("education", "") or sections.get("academic", "") or sections.get("qualifications", "")

    roles = _extract_roles_from_experience(experience_text)[:max_roles]
    skills = _extract_skills_from_skills_section(skills_section_text)
    if use_noun_chunks_for_skills and skills_section_text and len(skills) < max_skills:
        doc = nlp(skills_section_text[:3000])
        for chunk in doc.noun_chunks:
            c = _normalize(chunk.text)
            if 2 <= len(c) <= 60 and c.lower() not in {s.lower() for s in skills}:
                skills.append(c)
        skills = _unique_lower(skills, min_len=2)[:max_skills]
    education = _extract_education_phrases(education_text)
    if education_text:
        edu_doc = nlp(education_text[:2000])
        for ent in edu_doc.ents:
            if ent.label_ == "ORG":
                inst = _normalize(ent.text)
                if len(inst) >= 3 and inst.lower() not in {e.lower() for e in education}:
                    education.append(inst)
        education = _unique_lower(education, min_len=3)

    NON_LOCATION_WORDS = {
        "python", "java", "sql", "aws", "docker", "kubernetes", "react", "node",
        "excel", "word", "outlook", "access", "powerpoint", "windows", "linux",
        "agile", "scrum", "api", "rest", "graphql", "machine learning", "ml", "ai",
    }
    doc = nlp(text[:8000])
    locations = []
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC", "FAC"):
            tok = _normalize(ent.text)
            if tok.lower() in NON_LOCATION_WORDS:
                continue
            locations.append(tok)
    locations = _unique_lower(locations, min_len=2)

    return {
        "roles": roles[:max_roles],
        "skills": skills[:max_skills],
        "education": education,
        "locations": locations,
    }


def parse_resume_file(file_path: str, **kwargs) -> Dict[str, List[str]]:
    """Load resume from file (PDF/DOCX), extract text, then run parse_resume_ner."""
    from resume_parser_util import extract_text_from_file
    text = extract_text_from_file(file_path)
    return parse_resume_ner(text, **kwargs)


if __name__ == "__main__":
    import sys
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
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Parsing file: {path}")
        result = parse_resume_file(path)
    else:
        print("Parsing sample text (pass a file path to parse a file instead).")
        result = parse_resume_ner(SAMPLE)
    print("\n--- Extracted tags ---")
    for key in ["roles", "skills", "education", "locations"]:
        print(f"{key}: {result[key]}")
