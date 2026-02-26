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

# 2. SESSION STATE INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_context" not in st.session_state:
    st.session_state.last_context = {}


# 3. SIDEBAR (Combined implementation)
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
                
                # Call your existing utility
                text = extract_text_from_file(temp_path)
                
                # Store in session state
                st.session_state.extracted_resume_text = text
                st.session_state.current_file_name = uploaded_file.name
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                status.update(label="Resume Processed!", state="complete")
                st.success(f"Loaded: {uploaded_file.name}")
    
    st.divider()
    
    # Section: Manual Filters
    st.header("Manual Preferences")
    location_input = st.text_input("Preferred Location", placeholder="e.g. New York")
    experience_level = st.selectbox("Experience Level", 
                                    ["No Preference", "Entry level", "Associate", "Mid-Senior level", "Director"])

    st.divider()

    # Section: Advanced toggles.
    with st.expander("AI Search Settings"):
        ner_on = st.toggle("NER Keyword Extraction", value=True)
        llm_on = st.toggle("LLM Semantic Boosting", value=True)
    
    st.divider()
    
    # Section: Actions
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_results = None
        st.session_state.last_context = {}
        st.rerun()

# 4. MAIN UI HEADER
st.title("AI Career Assistant")
st.markdown(
    "Upload your resume and chat with the assistant to discover roles that best match your profile. "
    "You can ask for specific jobs or ask for career advice based on your matches."
)

# 5. DISPLAY CHAT HISTORY
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

# 6. INPUT AND LOGIC

# --- FLOATING MODE SELECTOR ---
# Create a container that stays at the bottom of the results but above the input
ui_container = st.container()

with ui_container:
    cols = st.columns([1, 3])
    with cols[0]:
        search_mode = st.toggle("🔍 **Search Mode**", value=True)
    
    if search_mode:
        st.caption("✨ Assistant will **search for jobs**.")
    else:
        st.caption("💬 Assistant is in **Career Coach** mode.")

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
        elif search_mode:
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
                
                # Format Display
                if results and results.get("ids") and results["ids"][0]:
                    intent_title = context.get("title", "relevant roles")
                    intent_loc = location_input if location_input else context.get("location", "your area")
                    
                    header_msg = f"I've found some **{intent_title}** in **{intent_loc}** that match your profile:"
                    st.markdown(header_msg)
                    
                    # PREPARE DATA FOR HISTORY: We extract exactly what we need for the UI
                    results_to_store = []
                    for i in range(len(results["ids"][0])):
                        meta = results["metadatas"][0][i]
                        job_data = {
                            "title": meta.get('title', 'Job'),
                            "company": meta.get('company', 'Company'),
                            "location": meta.get('location', 'N/A'),
                            "work_type": meta.get('work_type', 'N/A'),
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
    
    # Optional: Trigger a rerun to lock the state and keep UI synced
    st.rerun()

# to run: python -m streamlit run scripts/app.py