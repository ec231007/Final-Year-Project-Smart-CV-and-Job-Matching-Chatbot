import streamlit as st
import os
import json
from search_script import smart_search_with_file, collection
from career_agent import get_chatbot_response
from resume_parser_util import extract_text_from_file
from resume_ner_bert_v2 import _get_pipeline

# 0. SETUP PATHS & LOAD CACHE
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, "metadata_cache.json")

# --- WARMING UP MODELS TO REDUCE LATENCY---
@st.cache_resource(show_spinner=False)
def prewarm_models():
    """Forces ML models to load into memory on app start instead of on first search."""
    # 1. Wake up the NER pipeline
    _get_pipeline()

    # 2. Wake up ChromaDB and the Embedding Model by forcing a lightweight call
    collection.count() 

    return True

# Run the warm-up once per server start.
prewarm_models()
# ----------------------------


@st.cache_data # Cache this so we don't reload the JSON on every click
def load_metadata():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            data = json.load(f)
            return data.get("locations", [])
    return []

UNIQUE_LOCATIONS = load_metadata()

# --- 1. SESSION STATE ---
# This "basket" holds our actual selected locations
if "loc_basket" not in st.session_state:
    st.session_state.loc_basket = []

# --- 2. THE CUSTOM COMPONENT ---
def location_multiselect_custom():
    st.subheader("Location")
    
    # CSS for the 'Pills' (Selected Tags)
    st.markdown("""
        <style>
        .pills-container { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px; }
        .pill { 
            background-color: #e0e0e0; border-radius: 15px; padding: 2px 10px; 
            font-size: 0.8rem; display: flex; align-items: center; 
        }
        </style>
    """, unsafe_allow_html=True)

    if isinstance(st.session_state.loc_basket, str):
        st.session_state.loc_basket = [st.session_state.loc_basket]

    # 1. Render the 'Basket' as Pills
    if st.session_state.loc_basket:
        # We use a container to show the tags visually
        # Since Streamlit buttons can't easily sit inside a flexbox, 
        # we'll use a clean list with individual "X" buttons in a better layout
        cols = st.columns([0.9, 0.1])
        for loc in st.session_state.loc_basket:
            with st.container(border=True):
                c1, c2 = st.columns([0.8, 0.2])
                c1.text(loc)
                if c2.button("✖", key=f"del_{loc}", help=f"Remove {loc}"):
                    st.session_state.loc_basket.remove(loc)
                    st.rerun()
    
    # 2. The Popover (The "Dropdown" from your image)
    with st.popover("Add Locations...", use_container_width=True):
        search_term = st.text_input("Search", placeholder="Type to filter...", label_visibility="collapsed", key="loc_search_input")
        
        # Filter logic
        display_list = UNIQUE_LOCATIONS
        if search_term:
            display_list = [l for l in UNIQUE_LOCATIONS if search_term.lower() in l.lower()]
        
        # Scrollable area
        st.markdown('<div style="max-height: 250px; overflow-y: auto; padding: 5px;">', unsafe_allow_html=True)
        
        # Checkbox loop
        for loc in display_list[:50]:
            # This is the key: value= is driven by the basket
            # If the AI adds it to the basket, the checkbox will be checked automatically
            checked = st.checkbox(
                loc, 
                value=(loc in st.session_state.loc_basket), 
                key=f"chk_{loc}"
            )
            
            # Update basket based on checkbox click
            if checked and loc not in st.session_state.loc_basket:
                st.session_state.loc_basket.append(loc)
                st.rerun()
            elif not checked and loc in st.session_state.loc_basket:
                st.session_state.loc_basket.remove(loc)
                st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Clear All", use_container_width=True):
            st.session_state.loc_basket = []
            st.rerun()

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
if "search_buffer" not in st.session_state:
    st.session_state.search_buffer = [] # Stores results 6-20 of the LATEST search

def trim_history():
    """Keeps only the last 5 messages in the session state."""
    if len(st.session_state.messages) > 5:
        st.session_state.messages = st.session_state.messages[-5:]

# Logic for initializing filters
for key in ["filter_location", "filter_exp", "filter_type"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "exp" in key or "type" in key else ""

# Widget keys (to avoid the modification error, we ensure they exist)
if "loc_basket" not in st.session_state: st.session_state.loc_basket = []
if "exp_widget" not in st.session_state: st.session_state.exp_widget = []
if "type_widget" not in st.session_state: st.session_state.type_widget = []

# set ui states
is_search_mode = st.session_state.get("mode_toggle", True)
ner_on = st.session_state.get("ner_toggle", True)
llm_on = st.session_state.get("llm_toggle", True)

manual_filters = {
        "location": st.session_state.loc_basket,
        "experience": st.session_state.exp_widget,
        "work_type": [REVERSE_WORK_TYPE_MAP.get(t, t) for t in st.session_state.type_widget]
    }

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
            # --- PREPARE FILTERS ---
            current_manual_filters = {
                "location": st.session_state.get("loc_basket", []),
                "experience": st.session_state.get("exp_widget", []),
                "work_type": [REVERSE_WORK_TYPE_MAP.get(t, t) for t in st.session_state.get("type_widget", [])]
            }

            # --- SEARCH PATHWAY ---
            with st.spinner("Analyzing CV and searching roles..."):                
                # Execute Search Pipeline
                results, context = smart_search_with_file(
                    resume_text, 
                    prompt, 
                    NER_applied=ner_on, 
                    LLM_applied=llm_on,
                    manual_filters=manual_filters
                )
                
                # Store results in state so the chatbot can "see" them later
                st.session_state.last_results = results
                st.session_state.last_context = context

                #UPDATE SIDEBAR WIDGETS
                intent = context.get("intent", {})
                
                # This is where we update the global state. 
                # Because we call st.rerun() at the end, the sidebar will redraw with these
                if "location" in intent:
                    new_locs = intent["location"]
                    if isinstance(new_locs, str): new_locs = [new_locs]
    
                    # Merge AI findings with existing UI selections (remove duplicates)
                    combined = list(set(st.session_state.loc_basket + new_locs))
                    st.session_state.loc_basket = combined
                
                raw_exp = intent.get("experience", [])
                st.session_state.exp_widget = [raw_exp] if isinstance(raw_exp, str) else raw_exp
                
                raw_type = intent.get("work_type", [])
                if isinstance(raw_type, str): raw_type = [raw_type]
                st.session_state.type_widget = [WORK_TYPE_MAP.get(t, t) for t in raw_type if t in WORK_TYPE_MAP]
                
                # Format Display
                if results and results.get("ids") and results["ids"][0]:
                    intent_title = context.get("title", "relevant roles")
                    intent_loc = st.session_state.loc_basket if st.session_state.loc_basket else "your area"
                    
                    header_msg = f"I've found some **{intent_title}** in **{intent_loc}** that match your profile:"
                    st.markdown(header_msg)
                    
                    # Process all results into job objects
                    all_found_jobs = []
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
                        all_found_jobs.append(job_data)
                    
                    # NEW LOGIC: Split results
                    # Only the first 5 are "visible" to the UI and the RAG Chatbot
                    visible_now = all_found_jobs[:5]
                    # The rest are stored in a hidden buffer for "Show More"
                    st.session_state.search_buffer = all_found_jobs[5:]

                    # Append to messages (RAG will only see 'visible_now')
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": header_msg, 
                        "type": "search_results",
                        "results_data": visible_now
                    })
                else:
                    error_msg = "I couldn't find strong matches. Try broadening your terms!"
                    st.warning(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "type": "text"})
            trim_history()
        # --- Advice Mode Logic ---
        else:            
            # Get AI Response with a "Thinking" status
            with st.chat_message("assistant"):
                with st.status("Consulting career database...", expanded=True) as status:
                    st.write("Analyzing your request...")
                    
                    full_response, tool_used = get_chatbot_response(
                        prompt, 
                        st.session_state.messages,
                        search_results=st.session_state.get("last_results"),
                        search_context=st.session_state.get("last_context")
                    )
                    
                    if tool_used:
                        st.write("✅ Found relevant job trends. Synthesizing advice...")
                    
                    status.update(label="Advice ready!", state="complete", expanded=False)

                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            trim_history()

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
    location_multiselect_custom()
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

# 6. DISPLAY CHAT HISTORY
for idx, message in enumerate(st.session_state.messages):
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
            
            # --- SHOW MORE BUTTON ---
            # Only show if this is the last message and we have a buffer
            is_latest = (idx == len(st.session_state.messages) - 1)
            if is_latest and st.session_state.get("search_buffer"):
                if st.button(f"Show 5 more results ({len(st.session_state.search_buffer)} left)", key=f"more_btn_{idx}"):
                    next_five = st.session_state.search_buffer[:5]
                    st.session_state.search_buffer = st.session_state.search_buffer[5:]
                    message["results_data"].extend(next_five)
                    st.rerun()

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

# to run: python -m streamlit run scripts/app.py