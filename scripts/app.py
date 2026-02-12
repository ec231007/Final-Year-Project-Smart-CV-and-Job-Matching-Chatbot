import streamlit as st
import os
from search_script import smart_search_with_file

# 1. PAGE CONFIG
st.set_page_config(page_title="Smart CV Matcher", layout="wide")
st.title("AI Career Assistant")
st.markdown(
    "Upload your resume and chat with the assistant to discover roles that best match your profile."
)

# 2. SIDEBAR - Upload and Static Filters
with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Upload your Resume", type=['pdf', 'docx'])
    
    st.divider()
    st.header("Manual Preferences")
    # These can act as overrides if the AI misses something
    location_input = st.text_input("Preferred Location", placeholder="e.g. New York")
    experience_level = st.selectbox("Experience Level", 
                                   ["No Preference", "Entry level", "Associate", "Mid-Senior level", "Director"])

# 3. CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. ACTION
if prompt := st.chat_input("Ask me to find jobs or analyze your CV..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process search
    with st.chat_message("assistant"):
        if uploaded_file is None:
            st.warning("Please upload a resume in the sidebar first!")
        else:
            # Save file temporarily to process it
            with open("temp_resume", "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Analyzing your CV and searching for matching roles..."):
                results, intent = smart_search_with_file("temp_resume", prompt)

            # Handle cases where no results were found
            if not results or not results.get("ids") or not results["ids"][0]:
                title = intent.get("title") if intent else None
                location = intent.get("location") if intent else None
                summary_bits = []
                if title:
                    summary_bits.append(f"**{title}**")
                if location:
                    summary_bits.append(f"in **{location}**")
                summary = " ".join(summary_bits) if summary_bits else "your preferences"

                st.markdown(
                    f"I've analyzed {summary}, but I couldn't find strong matches in the database. "
                    "Try broadening your query, adjusting location, or relaxing experience level."
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "I analyzed your request but couldn't find strong matches. Try a broader query or different filters.",
                    }
                )
            else:
                # Format and Display Results
                intent_title = intent.get("title") if intent else None
                intent_location = intent.get("location") if intent else None
                title_text = intent_title or "roles"
                location_text = intent_location or "your preferred locations"

                response = (
                    f"I've analyzed your intent and I'm looking for **{title_text}** "
                    f"in **{location_text}**.\n\nHere are some roles that match your profile:"
                )
                st.markdown(response)

                for i in range(len(results["ids"][0])):
                    meta = results["metadatas"][0][i]
                    job_title = meta.get("title", "Job")
                    company = meta.get("company", "Company")
                    location = meta.get("location", "N/A")
                    work_type = meta.get("work_type", "N/A")
                    description = results["documents"][0][i][:500] + "..."

                    with st.expander(f"🎯 {job_title} at {company}"):
                        st.write(f"**Location:** {location} | **Type:** {work_type}")
                        st.write(description)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "I found some matches for you based on your resume and query. Explore the roles above.",
                    }
                )