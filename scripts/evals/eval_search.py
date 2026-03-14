import json
import os
import chromadb
from chromadb.utils import embedding_functions

# --- SETUP ---
DB_PATH = "data/job_vector_db"
COLLECTION_NAME = "linkedin_jobs"
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(SCRIPT_DIR, "metadata_cache.json")

client = chromadb.PersistentClient(path=DB_PATH)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

with open(CACHE_PATH, "r") as f:
    META_CACHE = json.load(f)
    UNIQUE_LOCATIONS = META_CACHE.get("locations", [])

def get_fuzzy_locations(user_locs):
    if not user_locs: return []
    if isinstance(user_locs, str): user_locs = [user_locs]
    matched = []
    for u_loc in user_locs:
        matched.extend([loc for loc in UNIQUE_LOCATIONS if u_loc.lower() in loc.lower()])
    return list(set(matched))

def flatten_to_string(val):
    """Ensures input is a string, even if it's a list or None."""
    if val is None: return ""
    if isinstance(val, list): return " ".join(str(i) for i in val)
    return str(val)

def run_eval_search(entry, NER_applied=True, LLM_applied=True):
    intent = entry.get("intent", {})
    
    # 1. BUILD FILTERS
    filter_parts = []
    for field in ["experience", "work_type"]:
        val = intent.get(field)
        if val:
            # Chroma $in expects a list
            filter_parts.append({field: {"$in": val if isinstance(val, list) else [val]}})
    
    loc_val = intent.get("location")
    if loc_val:
        matched_locs = get_fuzzy_locations(loc_val)
        if matched_locs:
            filter_parts.append({"location": {"$in": matched_locs}})

    final_where = None
    if len(filter_parts) > 1: final_where = {"$and": filter_parts}
    elif len(filter_parts) == 1: final_where = filter_parts[0]

    # 2. BUILD RICH QUERY (The Booster) - Fixes the TypeError
    query_parts = []
    
    # Item 0: Title
    query_parts.append(flatten_to_string(intent.get("title")))
    
    # Item 1: LLM Boost
    if LLM_applied and entry.get("llm_boost"):
        query_parts.append(flatten_to_string(entry["llm_boost"]))
    
    # Item 2: NER Tags
    if NER_applied and entry.get("ner_tags"):
        query_parts.append(flatten_to_string(entry["ner_tags"]))
    
    # Remove empty strings and join
    rich_query = " ".join([q for q in query_parts if q.strip()]).strip()

    # 3. EXECUTE
    results = collection.query(
        query_texts=[rich_query],
        n_results=10, 
        where=final_where,
    )

    matches = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            matches.append({
                "job_id": results["ids"][0][i],
                "score": round((1 - results["distances"][0][i]), 4),
                "title": results["metadatas"][0][i]["title"],
                "company": results["metadatas"][0][i]["company"],
                "location": results["metadatas"][0][i]["location"],
                "description": results["documents"][0][i] 
            })

    return {
        "config": {"ner": NER_applied, "llm": LLM_applied},
        "query_used": rich_query,
        "results": matches
    }