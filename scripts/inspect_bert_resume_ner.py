"""
Inspect raw BERT resume NER output — all entity labels and their values.

Run this to see exactly what labels the model returns and how it segments text,
so we can map them correctly to our roles/skills/education/locations (or use
the model's own field names).
Usage: python scripts/inspect_bert_resume_ner.py
"""

from collections import defaultdict

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
    from transformers import pipeline

    model_name = "yashpwr/resume-ner-bert-v2"
    print(f"Loading {model_name}...")
    pipe = pipeline(
        "token-classification",
        model=model_name,
        aggregation_strategy="simple",
        device=-1,
    )

    # Show all labels the model knows (from config)
    if hasattr(pipe.model, "config") and hasattr(pipe.model.config, "id2label"):
        labels = set()
        for idx, name in pipe.model.config.id2label.items():
            # BIO tags like B-Skills, I-Skills -> show base name once
            base = name.replace("B-", "").replace("I-", "").strip() if name != "O" else name
            labels.add(base)
        print("\n--- All labels in model config (id2label) ---")
        for L in sorted(labels):
            if L != "O":
                print(f"  {L}")
    else:
        print("(Could not read model id2label)")

    # Run on sample; chunk like the real parser (model uses max_length 128)
    text = " ".join(SAMPLE.split())
    tokenizer = pipe.tokenizer
    max_tokens, overlap = 120, 30
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(tokens) <= max_tokens:
        chunks = [text]
    else:
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
            start = end - overlap
            if start >= len(tokens):
                break

    raw = []
    for ci, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        out = pipe(chunk)
        for item in out or []:
            item["_chunk"] = ci + 1
            raw.append(item)

    print("\n--- Raw pipeline output (every entity) ---")
    if not raw:
        print("  (no entities)")
    else:
        for i, item in enumerate(raw):
            label = item.get("entity_group") or item.get("entity") or "?"
            word = item.get("word", "")
            score = item.get("score", 0)
            ch = item.get("_chunk", "")
            print(f"  [{i+1}] {label!r}: {word!r}  (score={score:.3f})  chunk={ch}")

    # Group by label
    by_label = defaultdict(list)
    for item in raw or []:
        label = item.get("entity_group") or item.get("entity") or "O"
        word = (item.get("word") or "").strip()
        if label == "O" or not word:
            continue
        by_label[label].append(word)

    print("\n--- Grouped by label (what we'd use for each column) ---")
    for label in sorted(by_label.keys()):
        vals = by_label[label]
        print(f"\n  {label}:")
        for v in vals:
            print(f"    - {v!r}")


if __name__ == "__main__":
    main()
