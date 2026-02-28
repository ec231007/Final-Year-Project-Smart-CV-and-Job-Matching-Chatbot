import json
import os
import chromadb
from chromadb.utils import embedding_functions
from groq_prompter import get_filter_json, get_search_query_llm
from resume_parser_util import extract_text_from_file
from resume_ner_bert import parse_resume_ner_bert as parse_resume_ner

# 1. SETUP PATHS & CONFIG
DB_PATH = "data/job_vector_db"
COLLECTION_NAME = "linkedin_jobs"
# Useing the same cache we generated in metadata_cache_db.py to help with location matching
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, "metadata_cache.json")

# 2. INITIALIZE CHROMA & CACHE
client = chromadb.PersistentClient(path=DB_PATH)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

# Fast-load metadata vocabulary from JSON
with open(CACHE_PATH, "r") as f:
    META_CACHE = json.load(f)
    UNIQUE_LOCATIONS = META_CACHE.get("locations", [])

def get_fuzzy_locations(user_loc):
    """Finds existing DB tags that contain the user's location string."""
    if not user_loc: return []
    return [loc for loc in UNIQUE_LOCATIONS if user_loc.lower() in loc.lower()]

# 3. THE SMART SEARCH PIPELINE
def smart_search_with_file(resume_text, additional_query="", NER_applied=True, LLM_applied=True):
    """
    Run the smart search pipeline on a resume file plus an optional free-text query.
    Returns a tuple of (results_dict_or_None, intent_dict).
    """

    print(f"Resume Text: {resume_text[:500]}") # Debug: Show the start of the resume text
    print(f"Additional Query: {additional_query}") # Debug: Show the additional query   
    print(f"NER Applied: {NER_applied}, LLM Applied: {LLM_applied}") # Debug: Show which features are applied

    # STEP A: Get Intent via Groq
    # We pass both the resume (for skills) and query (for specific filters)
    combined_input = f"RESUME: {resume_text[:2000]}\nUSER PREFERENCES: {additional_query}"
    intent = get_filter_json(combined_input)
    print(f"Extracted Intent: {intent}")

    # STEP B: Build Chroma Filter using Cache
    final_where = {}
    filter_parts = []

    # 1. Handle Experience (Multi-select)
    exp = intent.get("experience")
    if exp:
        # If it's a list (even with 1 item), use $in
        if isinstance(exp, list) and len(exp) > 0:
            filter_parts.append({"experience": {"$in": exp}})
        # If it's just a single string, you can use direct match or wrap it
        elif isinstance(exp, str):
            filter_parts.append({"experience": exp})

    # 2. Handle Work Type (Multi-select)
    wt = intent.get("work_type")
    if wt:
        if isinstance(wt, list) and len(wt) > 0:
            filter_parts.append({"work_type": {"$in": wt}})
        elif isinstance(wt, str):
            filter_parts.append({"work_type": wt})

    # 3. Handle Location (Fuzzy/String)
    raw_loc = intent.get("location")
    if raw_loc:
        # try match with locations in db
        matched_db_locations = get_fuzzy_locations(raw_loc)
        print(f"📍 Fuzzy Match: '{raw_loc}' mapped to {matched_db_locations}")
    
    if matched_db_locations:
        filter_parts.append({"location": {"$in": matched_db_locations}})
    else:
        # If no match found in DB, don't add a hard filter (it would return 0)
        print(f"⚠️ No exact DB match for {raw_loc}. Moving to semantic search.")

    # Combine parts into final_where
    final_where = None
    if len(filter_parts) > 1:
        final_where = {"$and": filter_parts}
    elif len(filter_parts) == 1:
        final_where = filter_parts[0]

    # STEP C: Build the "Rich Query" (The Booster Logic)
    # 1. Start with the basic title or user query
    base_query = intent.get("title") or additional_query or ""
    
    boost_parts = [base_query]

    # STEP D: Add LLM Semantic Summary (High Level Reasoning)
    if LLM_applied:
        llm_query = get_search_query_llm(resume_text, additional_query)
        boost_parts.append(llm_query)
        print(f"🤖 LLM Boost: {llm_query}")

    # Add NER Tags (Granular Keywords)
    if NER_applied:
        ner_output = parse_resume_ner(resume_text)
        # Extract and clean tags longer than 2 chars
        ner_tags = [s for k in ner_output for s in ner_output[k] if len(s) > 2]
        if ner_tags:
            boost_parts.append(" ".join(ner_tags))
            print(f"🏷️ NER Boost: {len(ner_tags)} tags added.")

    # Join everything into one big semantic string
    rich_query = " ".join(boost_parts)

    # STEP E: Query Database
    results = collection.query(
        query_texts=[rich_query],
        n_results=5,
        where=final_where,
    )

    # STEP F: Output Results (for debugging / CLI use)
    print(f"\n{'='*60}\n🔍 MATCHES FOR YOUR PROFILE\n{'='*60}")
    if not results["ids"][0]:
        print("No matches found with these filters. Try broader criteria.")
        # Still return the intent so the caller can surface it in the UI
        return None, intent

    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        score = round((1 - results["distances"][0][i]) * 100, 2)

        print(f"[{i+1}] {meta['title'].upper()} @ {meta['company']}")
        print(f"    📍 {meta['location']} | {meta['work_type']} | Match: {score}%")
        print(f"    📝 {results['documents'][0][i][:160]}...\n")

    search_context = {
        "intent": intent,
        "ner_tags": ner_tags if NER_applied else None,
        "llm_boost_query": llm_query if LLM_applied else None,
        "final_filters": final_where,
        "raw_resume_text": resume_text[:2000]
    }

    # Return both the raw results and the parsed intent for use in the frontend
    return results, search_context


# 4. OPTIONAL: CLI TEST ENTRYPOINT
if __name__ == "__main__":
    # Example: Pass a PDF/Doc and a specific location constraint
    test_file = extract_text_from_file(r"C:\Vasanth\Important stuff\Resumes\Vasanth Subramanian Resume.pdf")
    smart_search_with_file(test_file, "Software Engineer in New York", NER_applied=False, LLM_applied=False)
    smart_search_with_file(test_file, "Software Engineer in New York", NER_applied=True, LLM_applied=True)