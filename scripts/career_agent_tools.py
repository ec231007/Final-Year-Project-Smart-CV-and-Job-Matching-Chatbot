import chromadb
from chromadb.utils import embedding_functions

def query_career_advice_db(query, job_title=None, experience_level=None, n_results=5):
    """
    Advanced query tool that searches the job descriptions database to find 
    what real employers are looking for.
    """
    # 0. SETUP PATHS & CONFIG
    DB_PATH = "data/job_vector_db"
    COLLECTION_NAME = "linkedin_jobs"
    
    client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

    # 1. Build a smarter semantic query
    # Instead of a strict 'where' filter on title (which often fails due to exact-match rules), 
    # we prepend the job title to the query to give it massive semantic weight.
    semantic_query = query
    if job_title:
        semantic_query = f"Job Title: {job_title}. Core skills and requirements: {query}"

    # 2. Safely build the hard filter (only filter by experience if provided)
    where_filter = None
    if experience_level:
        where_filter = {"experience": experience_level}

    try:
        results = collection.query(
            query_texts=[semantic_query],
            n_results=n_results,
            where=where_filter
        )

        formatted_output = []
        if results and results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                doc_text = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                
                # Fetch correct metadata from the linkedin_jobs schema
                title = metadata.get("title", "Unknown Title")
                company = metadata.get("company", "Unknown Company")
                exp = metadata.get("experience", "Unknown Experience")
                
                snippet = doc_text[:600].replace("\n", " ").strip()
                
                formatted_output.append({
                    "job_title": title,
                    "company": company,
                    "experience": exp,
                    "snippet": f"{snippet}..."
                })
        return formatted_output
    except Exception as e:
        print(f"Database Query Error: {e}")
        return []

# --- Test Runner ---
if __name__ == "__main__":
    test_cases = [
        {
            "name": "Case 1: Broad Technical Search",
            "query": "Key skills including frameworks and databases",
            "job_title": "Python Developer",
            "experience": None
        },
        {
            "name": "Case 2: Targeted Search with Experience",
            "query": "Machine Learning, Deep Learning, and Statistics",
            "job_title": "Data Scientist",
            "experience": "Mid-Senior level" 
        },
        {
            "name": "Case 3: Soft Skills / Management",
            "query": "Stakeholder management and project lifecycle methodologies",
            "job_title": "Project Manager",
            "experience": None
        }
    ]

    print("--- STARTING TOOL TESTS ---\n")
    for case in test_cases:
        print(f"RUNNING: {case['name']}")
        print(f"Query: '{case['query']}' | Title: {case['job_title']} | Exp: {case['experience']}")
        
        try:
            results = query_career_advice_db(
                query=case['query'], 
                job_title=case['job_title'],
                experience_level=case['experience']
            )
            
            if not results:
                print("Result: [EMPTY] - No matches found.")
            else:
                for idx, res in enumerate(results):
                    print(f"  [{idx+1}] {res['job_title']} @ {res['company']} ({res['experience']})")
                    print(f"      Snippet: {res['snippet']}\n")
        except Exception as e:
            print(f"Result: [ERROR] - {str(e)}")
        
        print("-" * 50)