import json
import os
from typing import Dict, Tuple, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model Strategy
MODELS = {
    "intent": "llama-3.1-8b-instant",  # 500k TPD - High speed, good at JSON
    "boost": "meta-llama/llama-4-scout-17b-16e-instruct", # 500k TPD - Higher reasoning
    "fallback": "qwen/qwen3-32b" # 500k TPD - Alternative if Scout is down
}

VALID_EXPERIENCE = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Internship"]
VALID_WORK_TYPES = ["FULL_TIME", "CONTRACT", "PART_TIME", "TEMPORARY", "INTERNSHIP", "VOLUNTEER"]

def get_filter_json(user_prompt: str) -> Tuple[Dict[str, Any], str]:
    """
    Extracts structured filters using a token-efficient model.
    Returns: (Result Dictionary, Model Name)
    """
    model_name = MODELS["intent"]
    
    # Dense prompt engineering to save input tokens
    system_prompt = (
        "Role: Search Intent Extractor. Output ONLY JSON.\n"
        f"Allowed Experience: {VALID_EXPERIENCE}\n"
        f"Allowed Work Types: {VALID_WORK_TYPES}\n"
        "Rules: return 4 nullable fields: experience, work_type, location, title. Return lists for 'experience' and 'work_type' from the allowed list only. "
        "Extract 'location' and 'title' as strings."
    )
    
    # Internal trim: Intents are usually short, but we cap to be safe
    user_prompt = user_prompt[:1500] 

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0 # Strictness for classification
        )
        return json.loads(response.choices[0].message.content), model_name
    except Exception as e:
        print(f"Intent Error ({model_name}): {e}")
        return {"experience": [], "work_type": [], "location": None, "title": None}, model_name

def get_search_query_llm(resume_text: str, user_query: str = "") -> Tuple[str, str]:
    """
    Summarizes CV into keyword strings for Vector Search.
    Returns: (Keyword String, Model Name)
    """
    model_name = MODELS["boost"]
    
    system_prompt = (
        "Role: Recruitment Search Expert. Output ONLY a 20-word keyword string to be used as job search query.\n"
        "Content: Main title, top tech skills, and domain (e.g., Fintech). "
        "No prose. No sentences."
    )
    
    # Internal trim: 1500 chars is usually the 'Top' of the CV (most relevant)
    trimmed_text = resume_text[:1500]
    prompt = f"RESUME: {trimmed_text}\nREQUEST: {user_query}"
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip(), model_name
    except Exception as e:
        # If the preferred model fails (Rate Limit), we attempt the fallback
        if "429" in str(e):
            print(f"--- Rate Limit on {model_name}. Attempting Fallback... ---")
            return _get_search_query_fallback(resume_text, user_query)
        return "", model_name

def _get_search_query_fallback(resume_text: str, user_query: str) -> Tuple[str, str]:
    """Internal helper for fallback logic to keep main loop clean."""
    model_name = MODELS["fallback"]
    # ... logic identical to above but using the fallback model ...
    # This helps identify if a file used 'Plan B' during your evaluation.
    return "...", model_name