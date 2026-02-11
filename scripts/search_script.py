import json
import os
import chromadb
from chromadb.utils import embedding_functions
from groq_prompter import get_filter_json
from resume_parser_util import extract_text_from_file

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
def smart_search_with_file(file_path, additional_query=""):
    # STEP A: Extract Resume Text
    print(f"Processing: {os.path.basename(file_path)}")
    resume_text = extract_text_from_file(file_path)
    
    # STEP B: Get Intent via Groq
    # We pass both the resume (for skills) and query (for specific filters)
    combined_input = f"RESUME: {resume_text[:2000]}\nUSER PREFERENCES: {additional_query}"
    intent = get_filter_json(combined_input)
    
    # STEP C: Build Chroma Filter using Cache
    where_clauses = []
    
    # Standard exact filters
    if intent.get('experience'):
        where_clauses.append({"experience": intent['experience']})
    if intent.get('work_type'):
        where_clauses.append({"work_type": intent['work_type']})
    
    # Fuzzy Location Expansion (Matches user "NYC" to "New York, NY" from cache)
    if intent.get('location'):
        loc_variations = get_fuzzy_locations(intent['location'])
        if loc_variations:
            where_clauses.append({"location": {"$in": loc_variations}})

    # Combine into final 'where' dict
    final_where = None
    if len(where_clauses) > 1:
        final_where = {"$and": where_clauses}
    elif len(where_clauses) == 1:
        final_where = where_clauses[0]

    # STEP D: Query Database
    search_term = intent.get('title') or additional_query or "Job Opportunity"
    results = collection.query(
        query_texts=[search_term],
        n_results=5,
        where=final_where
    )

    # STEP E: Output Results
    print(f"\n{'='*60}\n🔍 MATCHES FOR YOUR PROFILE\n{'='*60}")
    if not results['ids'][0]:
        print("No matches found with these filters. Try broader criteria.")
        return

    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        score = round((1 - results['distances'][0][i]) * 100, 2)
        
        print(f"[{i+1}] {meta['title'].upper()} @ {meta['company']}")
        print(f"    📍 {meta['location']} | {meta['work_type']} | Match: {score}%")
        print(f"    📝 {results['documents'][0][i][:160]}...\n")

# 4. RUN IT
# Example: Pass a PDF/Doc and a specific location constraint
test_file = r"C:\Vasanth\Important stuff\Resumes\Vasanth Subramanian Resume.pdf" 
smart_search_with_file(test_file, "Software Engineer in New York")