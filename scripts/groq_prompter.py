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
    
    HARD CATEGORIES (Must match one of these or be null):
    - experience: {VALID_EXPERIENCE}
    - work_type: {VALID_WORK_TYPES}
    
    FUZZY CATEGORIES (Extract the name/term the user mentioned):
    - location (e.g., "NYC", "London", "Remote")
    - title (e.g., "Python Developer")
    - company (e.g., "Google")

    Return ONLY JSON. If a filter is not mentioned, use null.
    Example: "Junior dev in New York" -> {{"experience": "Entry level", "location": "New York", "title": "dev"}}
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

# Test it
print(get_filter_json("I want on-site junior roles in New York"))

def explain_matches(user_resume_text, job_results):
    # job_results comes from collection.query()
    
    prompt = f"""
    Compare this candidate's Resume to these Job Results.
    Explain WHY they matched and what they are missing for the top match.
    
    RESUME: {user_resume_text[:2000]} # Truncate to save tokens
    
    JOBS FOUND: {job_results['documents'][0]}
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content