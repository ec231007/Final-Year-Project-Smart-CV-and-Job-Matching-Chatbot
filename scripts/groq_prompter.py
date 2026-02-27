import json
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

VALID_EXPERIENCE = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Internship"]
VALID_WORK_TYPES = ["FULL_TIME", "CONTRACT", "PART_TIME", "TEMPORARY", "INTERNSHIP", "VOLUNTEER"]

def get_filter_json(user_prompt):
    system_prompt = f"""
    You are a Search Intent Extractor. Extract filters from the user's request.

    RULES:
    1. Extract multiple values if the user implies a range (e.g., "Junior or Mid-level").
    2. Only use values from the provided HARD CATEGORIES.
    3. Return lists for experience and work_type, even if there is only one value, Could have multiple values.
    
    HARD CATEGORIES (Must match one of these or be null):
    - experience: {VALID_EXPERIENCE}
    - work_type: {VALID_WORK_TYPES}
    
    FUZZY CATEGORIES (Extract the name/term the user mentioned):
    - location (e.g., "NYC", "London", "Remote")
    - title (e.g., "Python Developer")

    Return ONLY JSON. 
    Example: "Internships or entry level dev roles in London" 
    -> {{"experience": ["Entry level", "Internship"], "work_type": ["INTERNSHIP", "Full Time"], "location": "London", "title": "dev"}}
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"} # Forces the model to give clean JSON
    )
    return json.loads(response.choices[0].message.content)

def get_search_query_llm(resume_text, user_query=""):
    """
    Summarizes a CV and user intent into a condensed string of 
    searchable keywords for Vector DB retrieval.
    """
    system_prompt = """
    You are a Recruitment Search Expert. 
    Analyze the provided CV text and the user's specific request.
    Generate a condensed 20-30 word search query string that captures:
    1. The core job title/role.
    2. Primary technical skills (languages, frameworks, tools).
    3. Core industries or domain expertise (e.g., Fintech, AI, Backend).

    Edge cases:
    1. If the user query is empty/ not useful, build query from resume text.
    2. If both resume test and user query are empty/ not useful, return empty string "".

    Output ONLY the string of keywords, no introduction or JSON.
    Example Output: "Senior Python Developer AWS Docker Kubernetes Distributed Systems Fintech Scalability"
    """
    
    prompt = f"RESUME: {resume_text[:2500]}\nUSER REQUEST: {user_query}"
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1 # Low temperature for consistency
    )
    return response.choices[0].message.content.strip()