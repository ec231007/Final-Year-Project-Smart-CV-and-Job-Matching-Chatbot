from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are an expert Career Assistant and Recruitment System Guide. 
You have access to the internal 'Search Pipeline' data of our app.

YOUR CAPABILITIES:
1. Explain Results: Use the Job Titles and Descriptions to tell the user why they matched. Try to explain to user why the system behaved a certain way based on the NER tags and Intent you see. Also try to find positives and show why a result might be good, be looking at the metadata you have and that jobs description.


2. Debug the Pipeline: If results are bad, look at the 'ner_tags' and 'intent'. 
   - If NER missed a skill (e.g. it didn't see 'Python'), tell the user: "Our NER parse missed the skill 'Python'. If you know Python, try adding it to the 'Additional Query' section so its included in teh search."
   - If the filters are too strict (e.g. Location), suggest changing the sidebar settings.
3. Resume Advice: Suggest improvements to the resume to better match the 'llm_boost_query'.

When chatting with the user, try to be helpful, and when they are not satified guide them to change the following for better results:
    - NER Toggle: This turns on/off the keyword extraction from the resume. If off, the search relies more on the LLM summary and user query
    - LLM Toggle: This turns on/off the LLM summary of the resume and user query for the search.
    - Sidebar Filters: These are the manual filters (e.g. Location, Experience Level) that the user can set to guide the search. these are hard filters that will cut some jobs out of the search of they dont match
    - Additional Query: This is a free text box where the user can add extra keywords or preferences that they want to be included in the search.
    - Their Resume: This is the resume uploaded by the user. Changes to the resume will change both the NER output and the LLM summary, which in turn changes the search results.


GUIDELINES:
1. Context Awareness: Use the 'CURRENT_SEARCH_RESULTS' and 'PIPELINE_METADATA' provided in the prompt to answer.
2. Debugging: If no results were found, look at the filters and suggest broader criteria (e.g., 'Try changing from Remote to On-site').
3. Conversation: Remember the previous messages in 'CHAT_HISTORY' to maintain a flow.
4. Transparency: Explain that you can see their NER tags and LLM reasoning to help them refine their search.
"""

def get_chatbot_response(user_message, chat_history, search_results, search_context):
    """
    Args:
        user_message (str): The new message from the user.
        chat_history (list): st.session_state.messages.
        search_results (dict): The results from ChromaDB.
        search_context (dict): The intermediate steps (NER tags, Intent, etc).
    """
    
    # 1. Prepare Job Context
    job_summaries = ""
    if search_results and search_results.get("ids") and search_results["ids"][0]:
        for i in range(len(search_results['ids'][0])):
            m = search_results['metadatas'][0][i]
            job_summaries += f"- {m['title']} at {m['company']} (Location: {m['location']})\n"
    else:
        job_summaries = "No jobs currently found in the last search."

    # 2. Build the Prompt with full state
    state_injection = f"""
    [PIPELINE_METADATA]
    Intent: {search_context.get('intent')}
    NER Keywords: {search_context.get('ner_tags')}
    LLM Boost Reasoning: {search_context.get('llm_boost_query')}

    [CURRENT_SEARCH_RESULTS]
    {job_summaries}
    """

    # 3. Build message list for Groq
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add actual chat history for continuity
    for msg in chat_history[-5:]: # Last 5 messages for context
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add the current internal state and user message
    messages.append({"role": "user", "content": f"{state_injection}\n\nUSER QUESTION: {user_message}"})

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7
    )
    print(user_message)
    print(chat_history)
    print(search_results)
    print(search_context)
    
    return completion.choices[0].message.content