import chromadb
from chromadb.utils import embedding_functions

# 1. Setup - Use the EXACT same settings as the ingestion script
DB_PATH = "data/job_vector_db"
COLLECTION_NAME = "linkedin_jobs"

client = chromadb.PersistentClient(path=DB_PATH)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

def search_jobs(query_text, n_results=3):
    print(f"\nSearching for: '{query_text}'...")
    
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    # 2. Print the matches nicely
    for i in range(len(results['ids'][0])):
        print("-" * 50)
        print(f"MATCH #{i+1} (ID: {results['ids'][0][i]})")
        print(f"TITLE: {results['metadatas'][0][i]['title']}")
        print(f"COMPANY: {results['metadatas'][0][i]['company']}")
        print(f"LOCATION: {results['metadatas'][0][i]['location']}")
        # Snippet of the document
        print(f"PREVIEW: {results['documents'][0][i][:200]}...")

# 3. Test it!
search_jobs("Junior Python Developer with SQL experience")
search_jobs("Data Analyst in London")