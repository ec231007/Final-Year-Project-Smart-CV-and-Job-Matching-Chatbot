import streamlit as st
import os
from search_script import smart_search_with_file
from career_agent import get_chatbot_response
from resume_parser_util import extract_text_from_file

# 1. PAGE CONFIG
st.set_page_config(page_title="Smart CV Matcher", layout="wide")
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .job-card { padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; margin-bottom: 10px; }
    /* Style to make the mode selector look integrated */
    div[data-testid="stVerticalBlock"] > div:has(div.stToggle) {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: -20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Define constants for multi-select options
VALID_EXPERIENCE = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Internship"]
VALID_WORK_TYPES = ["FULL_TIME", "CONTRACT", "PART_TIME", "TEMPORARY", "INTERNSHIP", "VOLUNTEER"]
# Human-friendly labels for the UI
WORK_TYPE_MAP = {"FULL_TIME": "Full-time", "CONTRACT": "Contract", "PART_TIME": "Part-time", "TEMPORARY": "Temporary", "INTERNSHIP": "Internship", "VOLUNTEER": "Volunteer"}
# Reverse map for when the user selects something in the UI to send back to the Search Script
REVERSE_WORK_TYPE_MAP = {v: k for k, v in WORK_TYPE_MAP.items()}

# 2. SESSION STATE INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_context" not in st.session_state:
    st.session_state.last_context = {}

# Logic for initializing filters
for key in ["filter_location", "filter_exp", "filter_type"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "exp" in key or "type" in key else ""

# Widget keys (to avoid the modification error, we ensure they exist)
if "loc_widget" not in st.session_state: st.session_state.loc_widget = ""
if "exp_widget" not in st.session_state: st.session_state.exp_widget = []
if "type_widget" not in st.session_state: st.session_state.type_widget = []

# set ui states
is_search_mode = st.session_state.get("mode_toggle", True)
ner_on = st.session_state.get("ner_toggle", True)
llm_on = st.session_state.get("llm_toggle", True)

# 3. Processing Logic: We have two main pathways: Search Mode and Chatbot Mode. The toggle determines which one we take when the user submits a message.
if prompt := st.chat_input("Ask me to find jobs, or chat about your career..."):
    
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 2. VALIDATION: Ensure resume exists before any action
        # Use the session state text instead of checking the uploaded_file directly
        resume_text = st.session_state.get("extracted_resume_text")
        
        if resume_text is None:
            msg = "Please upload a resume in the sidebar first!"
            st.warning(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        
        # 3. EXECUTION LOGIC: Based strictly on the toggle
        elif is_search_mode:
            # --- SEARCH PATHWAY ---
            with st.spinner("Analyzing CV and searching roles..."):                
                # Execute Search Pipeline
                results, context = smart_search_with_file(
                    resume_text, 
                    prompt, 
                    NER_applied=ner_on, 
                    LLM_applied=llm_on
                )
                
                # Store results in state so the chatbot can "see" them later
                st.session_state.last_results = results
                st.session_state.last_context = context

                #UPDATE SIDEBAR WIDGETS
                intent = context.get("intent", {})
                
                # This is where we update the global state. 
                # Because we call st.rerun() at the end, the sidebar will redraw with these
                if "location" in intent:
                    st.session_state.loc_widget = intent["location"]
                
                raw_exp = intent.get("experience", [])
                st.session_state.exp_widget = [raw_exp] if isinstance(raw_exp, str) else raw_exp
                
                raw_type = intent.get("work_type", [])
                if isinstance(raw_type, str): raw_type = [raw_type]
                st.session_state.type_widget = [WORK_TYPE_MAP.get(t, t) for t in raw_type if t in WORK_TYPE_MAP]
                
                # Format Display
                if results and results.get("ids") and results["ids"][0]:
                    intent_title = context.get("title", "relevant roles")
                    intent_loc = st.session_state.loc_widget if st.session_state.loc_widget else "your area"
                    
                    header_msg = f"I've found some **{intent_title}** in **{intent_loc}** that match your profile:"
                    st.markdown(header_msg)
                    
                    # PREPARE DATA FOR HISTORY: We extract exactly what we need for the UI
                    results_to_store = []
                    for i in range(len(results["ids"][0])):
                        meta = results["metadatas"][0][i]
                        # Use the map to show "Full-time" instead of "FULL_TIME"
                        raw_wt = meta.get('work_type', 'N/A')
                        display_wt = WORK_TYPE_MAP.get(raw_wt, raw_wt)
                        job_data = {
                            "title": meta.get('title', 'Job'),
                            "company": meta.get('company', 'Company'),
                            "location": meta.get('location', 'N/A'),
                            "work_type": display_wt,
                            "snippet": results["documents"][0][i][:400] + "..."
                        }
                        results_to_store.append(job_data)
                        
                        # Display the card immediately
                        with st.expander(f"🎯 {job_data['title']} @ {job_data['company']}"):
                            st.write(f"**Location:** {job_data['location']} | **Type:** {job_data['work_type']}")
                            st.write(job_data['snippet'])
                    
                    # Append as a 'search_results' type
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": header_msg, 
                        "type": "search_results",
                        "results_data": results_to_store
                    })
                else:
                    error_msg = "I couldn't find strong matches for that specific query. Try broadening your terms!"
                    st.warning(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "type": "text"})
        
        else:
            # --- CHATBOT PATHWAY ---
            with st.spinner("Thinking..."):
                ai_response = get_chatbot_response(
                    prompt, 
                    st.session_state.messages, 
                    st.session_state.last_results, 
                    st.session_state.last_context
                )
                st.markdown(ai_response)
                # Append as a standard 'text' type
                st.session_state.messages.append({"role": "assistant", "content": ai_response, "type": "text"})

    # Trigger a rerun to lock the state and keep UI synced
    print(f"DEBUG: Saved to State -> {st.session_state.last_context.get('intent')}")
    st.rerun()

# 4. SIDEBAR FOR UPLOAD AND MANUAL FILTERS
with st.sidebar:
    st.title("Control Panel")
    
    # Section: Upload
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Upload your Resume", type=['pdf', 'docx'])

    if uploaded_file:
        # Check if we have already processed this specific file
        # We use the filename as a simple key
        if st.session_state.get("current_file_name") != uploaded_file.name:
            with st.status("Processing Resume...") as status:
                st.write("Extracting text...")
                # Temporary save to disk for the parser to read
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                text = extract_text_from_file(temp_path)
                st.session_state.extracted_resume_text = text
                st.session_state.current_file_name = uploaded_file.name
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                status.update(label="Resume Processed!", state="complete")
                st.success(f"Loaded: {uploaded_file.name}")
    
    st.divider()
    
    # Section: Manual Filters
    st.header("Manual Preferences")

    # If the Search Logic updates these keys, the sidebar "refreshes" automatically.
    st.text_input("Preferred Location", key="loc_widget")
    st.multiselect("Experience Level", options=VALID_EXPERIENCE, key="exp_widget")
    st.multiselect("Work Type", options=list(WORK_TYPE_MAP.values()), key="type_widget")
    
    st.divider()

    with st.expander("AI Search Settings"):
        st.toggle("NER Keyword Extraction", value=True, key="ner_toggle")
        st.toggle("LLM Semantic Boosting", value=True, key="llm_toggle")
    
    if st.button("Clear Conversation", use_container_width=True):
        for key in ["messages", "last_results", "last_context", "filter_location", "filter_exp", "filter_type"]:
            st.session_state[key] = [] if "filter" in key else {}
        st.rerun()


# 5. MAIN UI HEADER
st.title("AI Career Assistant")
st.markdown(
    "Upload your resume and chat with the assistant to discover roles that best match your profile. "
    "You can ask for specific jobs or ask for career advice based on your matches."
)
# --- FLOATING MODE SELECTOR ---
# Create a container that stays at the bottom of the results but above the input
ui_container = st.container()

with ui_container:
    cols = st.columns([1, 3])
    with cols[0]:
        st.toggle("🔍 **Search Mode**", value=True, key="mode_toggle", help="Switch between searching jobs and career advice.")
    
    if is_search_mode:
        st.caption("✨ Assistant will **search for jobs**.")
    else:
        st.caption("💬 Assistant is in **Career Coach** mode.")

# 6. DISPLAY CHAT HISTORY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Handle standard text messages (User or AI Coach)
        if message.get("type") == "text" or "type" not in message:
            st.markdown(message["content"])
        
        # Handle Search Results (Ensures they persist across reruns)
        elif message.get("type") == "search_results":
            st.markdown(message["content"]) # This is the "Found matches for..." header
            
            # Re-render the job cards from the stored data
            if "results_data" in message:
                for job in message["results_data"]:
                    with st.expander(f"🎯 {job['title']} @ {job['company']}"):
                        st.write(f"**Location:** {job['location']} | **Type:** {job['work_type']}")
                        st.write(job['snippet'])

# to run: python -m streamlit run scripts/app.py