import json
import time
import os
from tqdm import tqdm
import sys
# Ensure local imports work
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from career_agent import get_chatbot_response
from search_script import smart_search_with_file

INPUT_CV_DATA = "data/processed_eval_data.json"
OUTPUT_EVAL_DATA = "data/chat_eval_dataset.json"

def generate_curated_dataset():
    # 1. Load a small, diverse sample of resumes
    if not os.path.exists(INPUT_CV_DATA):
        print("Dataset not found. Please ensure processed_eval_data.json exists.")
        return
        
    with open(INPUT_CV_DATA, "r") as f:
        full_data = json.load(f)
    
    # Pick 10 resumes from different categories to ensure variety
    categories_seen = set()
    sample_data = []
    for entry in full_data:
        if entry["ground_truth_cat"] not in categories_seen:
            categories_seen.add(entry["ground_truth_cat"])
            sample_data.append(entry)
        if len(sample_data) == 10:
            break

    chat_records = []

    print(f"Generating Evals for {len(sample_data)} Resumes (10 cases each)...")

    for entry in tqdm(sample_data, desc="Processing Resumes"):
        cv_text = entry["resume_text"]
        target_cat = entry["ground_truth_cat"]
        
        # --- GENERATE REAL CONTEXT ---
        # Run a real search to get rich context (NER=True, LLM=False to save tokens)
        search_results, search_context = smart_search_with_file(
            resume_text=cv_text,
            additional_query=target_cat,
            NER_applied=True,
            LLM_applied=False 
        )
        
        # Get the top job title for the explanation test
        top_job_title = "a job"
        if search_results and search_results.get("ids") and search_results["ids"][0]:
            top_job_title = search_results["metadatas"][0][0].get("title", "a job")

        # --- DEFINE THE 5 TEST SCENARIOS ---
        scenarios = [
            {
                "type": "Match_Explanation",
                "prompt": f"Why did I match with the '{top_job_title}' role? What specific parts of my CV stood out?",
                "use_context": True,
                "expect_tool": False
            },
            {
                "type": "Tool_Advice_With_Context",
                "prompt": f"Based on my current matches, what hard skills should I learn to get better '{target_cat}' roles?",
                "use_context": True,
                "expect_tool": True
            },
            {
                "type": "System_Doubt",
                "prompt": "I noticed the NER tags in the system state. What happens if I turn the NER toggle off?",
                "use_context": True,
                "expect_tool": False
            },
            {
                "type": "Tool_Advice_No_Context",
                "prompt": "I'm thinking of pivoting to Data Science. What are the core skills employers want for that?",
                "use_context": False,
                "expect_tool": True
            },
            {
                "type": "General_Advice_No_Context",
                "prompt": "Can you act as a career coach and give me general tips for an interview?",
                "use_context": False,
                "expect_tool": False
            }
        ]

        # --- RUN THE SCENARIOS ---
        for scene in scenarios:
            # Setup Context
            s_results = search_results[:5] if scene["use_context"] else None
            s_context = search_context if scene["use_context"] else {}
            if s_results and "ids" in s_results:
                s_results = {
                    "ids": [s_results["ids"][0][:5]],
                    "metadatas": [s_results["metadatas"][0][:5]],
                    "documents": [s_results["documents"][0][:5]],
                    "distances": [s_results["distances"][0][:5]]
                }
            
            # Simulate Chat History (e.g., user just searched)
            chat_history = []
            if scene["use_context"] and s_results:
                chat_history = [{
                    "role": "assistant",
                    "content": f"I've found some {target_cat} roles that match your profile:",
                    "type": "search_results"
                }]

            try:
                # Call the Agent
                response_text, tool_used = get_chatbot_response(
                    user_message=scene["prompt"],
                    chat_history=chat_history,
                    search_results=s_results,
                    search_context=s_context
                )
                
                chat_records.append({
                    "filename": entry["filename"],
                    "resume_category": target_cat,
                    "scenario_type": scene["type"],
                    "user_prompt": scene["prompt"],
                    "agent_response": response_text,
                    "tool_used": tool_used,
                    "expected_tool": scene["expect_tool"],
                    "context_provided": bool(scene["use_context"])
                })
                
                # API Rate Limit Safety
                time.sleep(2)
                
            except Exception as e:
                print(f"\nError on {entry['filename']} - {scene['type']}: {e}")

    # Save Results
    with open(OUTPUT_EVAL_DATA, "w") as f:
        json.dump(chat_records, f, indent=4)
    print(f"\n✅ Successfully generated {len(chat_records)} curated test cases!")

if __name__ == "__main__":
    generate_curated_dataset()