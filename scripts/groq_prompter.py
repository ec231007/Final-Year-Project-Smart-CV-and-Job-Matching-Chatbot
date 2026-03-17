import json
import os
import time
from typing import Dict, Tuple, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model Strategy
MODELS = {
    "intent": "llama-3.1-8b-instant",  # Primary for filters
    "boost": "meta-llama/llama-4-scout-17b-16e-instruct", # Primary for summary
    "fallback": "qwen/qwen3-32b" # The "Plan B" for both
}

VALID_EXPERIENCE = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Internship"]
VALID_WORK_TYPES = ["FULL_TIME", "CONTRACT", "PART_TIME", "TEMPORARY", "INTERNSHIP", "VOLUNTEER"]

def get_filter_json(user_prompt: str, model_override: str = None) -> Tuple[Dict[str, Any], str]:
    """
    Extracts structured filters. If 429 occurs, tries the fallback model.
    """
    model_name = model_override or MODELS["intent"]
    
    system_prompt = (
        "Role: Search Intent Extractor. Output ONLY JSON.\n"
        f"Allowed Experience: {VALID_EXPERIENCE}\n"
        f"Allowed Work Types: {VALID_WORK_TYPES}\n"
        "Rules: return 4 nullable fields: experience, work_type, location, title. "
        "Return lists for 'experience' and 'work_type' from allowed list only."
        "If experience or work_type is not mentioned, return as many applicable [or the full lists]. As not having key catagories like 'FULL_TIME' or 'Mid-Senior level' can be stop results. For example, if no experience level is mentioned, return all experience levels. If no work type is mentioned, return all work types. Always return a location and title if mentioned, but if not mentioned, return null for those fields."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt[:1500]}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content), model_name

    except Exception as e:
        # Check for Rate Limit and trigger fallback if it's the first attempt
        if "429" in str(e) and model_name != MODELS["fallback"]:
            print(f"--- Intent Rate Limit on {model_name}. Trying Fallback {MODELS['fallback']} ---")
            return get_filter_json(user_prompt, model_override=MODELS["fallback"])
        
        print(f"Intent Error on {model_name}: {e}")
        return {"experience": [], "work_type": [], "location": None, "title": None}, model_name

def get_search_query_llm(resume_text: str, user_query: str = "", model_override: str = None) -> Tuple[str, str]:
    """
    Summarizes CV into keywords. If 429 occurs, tries the fallback model.
    """
    model_name = model_override or MODELS["boost"]
    
    system_prompt = (
        "Role: Recruitment Search Expert. Output ONLY a 20-word keyword string for job search.\n"
        "Content: Main title, top tech skills, and domain. No prose."
    )
    
    prompt = f"RESUME: {resume_text[:1500]}\nREQUEST: {user_query}"
    
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
        if "429" in str(e) and model_name != MODELS["fallback"]:
            print(f"--- Boost Rate Limit on {model_name}. Trying Fallback {MODELS['fallback']} ---")
            return get_search_query_llm(resume_text, user_query, model_override=MODELS["fallback"])
        
        print(f"Boost Error on {model_name}: {e}")
        return "", model_name