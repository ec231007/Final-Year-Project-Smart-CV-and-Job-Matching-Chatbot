"""
Inspect raw spaCy (en_core_web_sm) output on a sample resume.

Kept to show why we no longer use spaCy for resume NER: the general-purpose model
mis-tags resume text (e.g. Python→GPE, Java→PERSON, CA→PRODUCT). We use the
BERT resume NER model instead (see resume_ner_bert.py).
Run: python scripts/inspect_spacy_resume.py
"""

import spacy

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


def main():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(SAMPLE)

    print("=" * 60)
    print("1. NAMED ENTITIES (doc.ents)")
    print("   Label legend: PERSON, ORG, GPE, LOC, DATE, etc.")
    print("=" * 60)
    for ent in doc.ents:
        print(f"   [{ent.label_:<8}] {repr(ent.text)}")

    print("\n" + "=" * 60)
    print("2. NOUN CHUNKS (doc.noun_chunks)")
    print("=" * 60)
    for i, chunk in enumerate(doc.noun_chunks):
        print(f"   {i+1:2}. {repr(chunk.text)}")

    print("\n" + "=" * 60)
    print("3. TOKENS (first 80) — text | pos_ | dep_ | head.text")
    print("   Useful to see how 'Experience', 'Skills', 'Education' are tagged.")
    print("=" * 60)
    for i, tok in enumerate(doc):
        if i >= 80:
            print("   ... (truncated)")
            break
        print(f"   {tok.text!r:<25} {tok.pos_:<6} {tok.dep_:<10} head={tok.head.text!r}")

    print("\n" + "=" * 60)
    print("4. LINES THAT LOOK LIKE SECTION HEADERS")
    print("   (single-word or short lines that might be section titles)")
    print("=" * 60)
    for line in doc.text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Only show short lines (likely headers)
        if (len(line) <= 15) or (len(line) < 35 and " " in line):
            tokens = list(nlp(line))
            pos = " ".join(f"{t.text}/{t.pos_}" for t in tokens)
            print(f"   {repr(line):<40} -> {pos}")

    print("\n" + "=" * 60)
    print("5. ENTITY LABELS EXPLAINED (spaCy en_core_web_sm)")
    print("=" * 60)
    print("   PERSON = person names")
    print("   ORG    = organizations, companies")
    print("   GPE    = countries, cities, states")
    print("   LOC    = non-GPE locations")
    print("   DATE   = dates")
    print("   (No custom 'SKILL' or 'JOB_TITLE' — we use sections + patterns)")


if __name__ == "__main__":
    main()
