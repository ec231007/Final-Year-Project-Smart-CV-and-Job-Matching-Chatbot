import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from pathlib import Path
import os

# 1. SETUP PATHS DYNAMICALLY
# This finds the folder where this script lives (scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent
# This goes one level up to the project root (smart-cv-chatbot/)
PROJECT_ROOT = SCRIPT_DIR.parent

# Define input/output paths relative to root
CSV_PATH = PROJECT_ROOT / "data" / "LinkedIn" / "postings.csv"
DB_PATH = PROJECT_ROOT / "data" / "job_vector_db"

COLLECTION_NAME = "linkedin_jobs"

# 2. Initialize ChromaDB (Persistent on your laptop)
client = chromadb.PersistentClient(path=DB_PATH)

# Use the industry-standard lightweight model for embeddings
# This will download automatically on the first run (~80MB)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME, 
    embedding_function=emb_fn
)

# 3. Load and Clean Data
print("Loading CSV...")
# usecols helps save memory on your laptop by only loading what we need
cols_to_use = ['job_id', 'title', 'description', 'skills_desc', 'location', 
               'company_name', 'formatted_experience_level', 'work_type']
df = pd.read_csv(CSV_PATH, usecols=cols_to_use).fillna("")

# For testing on your laptop, maybe start with the first 1000 rows
# df = df.head(1000) 

print(f"Processing {len(df)} jobs...")

# 4. Ingestion Loop
# We batch things to make it faster and easier on your CPU
batch_size = 100
for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i : i + batch_size]
    
    documents = []
    metadatas = []
    ids = []
    
    for _, row in batch.iterrows():
        # This is the "Searchable Text" - we combine title, description, and skills
        combined_text = f"Title: {row['title']}\nLocation: {row['location']}\nSkills: {row['skills_desc']}\nDescription: {row['description']}"
        
        documents.append(combined_text)
        
        # Metadata allows us to 'Filter' later (e.g., "only London")
        metadatas.append({
            "title": str(row['title']),
            "location": str(row['location']),
            "company": str(row['company_name']),
            "experience": str(row['formatted_experience_level']),
            "work_type": str(row['work_type'])
        })
        
        ids.append(str(row['job_id']))
    
    # Add to the Vector Database
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

print(f"Success! Your Vector DB is ready at {DB_PATH}")