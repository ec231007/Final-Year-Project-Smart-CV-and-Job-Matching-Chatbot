import chromadb
from chromadb.utils import embedding_functions
from groq_prompter import get_filter_json

# Setup ChromaDB
DB_PATH = "data/job_vector_db"
COLLECTION_NAME = "linkedin_jobs"
client = chromadb.PersistentClient(path=DB_PATH)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

# In-memory cache of unique tags for fast fuzzy matching
print("Initializing metadata cache...")
all_metas = collection.get(include=['metadatas'])['metadatas']
unique_locations = list(set(m['location'] for m in all_metas if 'location' in m))

def get_fuzzy_locations(user_loc):
    if not user_loc: return []
    return [loc for loc in unique_locations if user_loc.lower() in loc.lower()]

def smart_search(user_query):
    # STEP A: Extract Intent via Groq
    intent = get_filter_json(user_query)
    
    # STEP B: Build Chroma Filter
    where_clauses = []
    if intent.get('experience'):
        where_clauses.append({"experience": intent['experience']})
    if intent.get('work_type'):
        where_clauses.append({"work_type": intent['work_type']})
    
    # Fuzzy Location Expansion
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

    # STEP C: Query Database
    # We query for n_results=5 as requested
    results = collection.query(
        query_texts=[intent.get('title') or user_query],
        n_results=5,
        where=final_where
    )

    # STEP D: Print the Results
    print(f"\n{'='*60}")
    print(f"SEARCH RESULTS FOR: {user_query}")
    print(f"AI FILTERS APPLIED: {intent}")
    print(f"{'='*60}")

    if not results['ids'][0]:
        print("No matches found with those specific filters.")
        return

    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        dist = results['distances'][0][i]
        # Lower distance = Better match (Distance of 0.0 is a perfect match)
        
        print(f"\n[{i+1}] {meta['title'].upper()}")
        print(f"📍 {meta['location']} | 🏢 {meta['company']}")
        print(f"📊 Level: {meta['experience']} | Type: {meta['work_type']}")
        print(f"🔗 Match Score: {round((1-dist)*100, 2)}%") 
        print(f"📝 Description: {results['documents'][0][i][:250]}...")
        print("-" * 30)

# 3. RUN IT
smart_search("Senior Analyst roles in London, preferably on-site")