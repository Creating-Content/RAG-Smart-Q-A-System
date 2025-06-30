# streamlit_app.py
import streamlit as st
import requests
import json
import os
import time # For brief pauses

# --- Configuration ---
FASTAPI_BASE_URL = "http://127.0.0.1:8000" # Ensure your FastAPI backend is running on this URL

st.set_page_config(layout="wide", page_title="Smart QA System")

st.title("🧠 Smart QA System")
st.markdown("Ask questions about your documents (PDF, CSV) and web links.") # Updated description

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper function for API calls ---
def call_fastapi(endpoint, method="POST", data=None, files=None):
    """
    Helper function to make requests to the FastAPI backend.
    Handles JSON data and file uploads.
    """
    url = f"{FASTAPI_BASE_URL}{endpoint}"
    try:
        if files:
            response = requests.request(method, url, files=files)
        elif data:
            response = requests.request(method, url, json=data)
        else:
            response = requests.request(method, url)

        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": f"Could not connect to FastAPI backend at {FASTAPI_BASE_URL}. Please ensure it is running."}
    except requests.exceptions.HTTPError as e:
        return {"status": "error", "message": f"API Error: {e.response.status_code} - {e.response.text}"}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}

# --- Ingestion Section ---
st.sidebar.header("Ingest New Knowledge")

with st.sidebar:
    # New: Ingestion Mode Selector
    st.subheader("Ingestion Mode")
    ingestion_mode = st.radio(
        "Choose how to ingest content:",
        ("Add to existing knowledge base", "Clear existing & ingest new"),
        key="ingestion_mode_radio"
    )

    if ingestion_mode == "Clear existing & ingest new":
        st.warning("Selecting this will clear ALL previously ingested data before adding new content.")
        # Optional: Add a button to manually clear DB if desired, independent of new ingest
        if st.button("Manually Clear Knowledge Base", key="manual_clear_db_btn"):
            with st.spinner("Clearing knowledge base..."):
                clear_result = call_fastapi("/clear_db", method="POST")
                if clear_result.get("status") == "error":
                    st.error(f"Failed to clear DB: {clear_result['message']}")
                else:
                    st.success("Knowledge base cleared!")
                    st.session_state.messages = [] # Clear chat history as well
                    st.rerun() # Rerun to reflect cleared state


    # Function to handle ingestion with mode selection
    def handle_ingestion(ingest_type, data_or_file, endpoint, success_message):
        if ingestion_mode == "Clear existing & ingest new":
            with st.spinner("Clearing previous knowledge..."):
                clear_result = call_fastapi("/clear_db", method="POST")
                if clear_result.get("status") == "error":
                    st.error(f"Failed to clear previous DB: {clear_result['message']}")
                    return # Stop if clearing failed
                else:
                    st.session_state.messages = [] # Clear chat history immediately
                    time.sleep(1) # Give a brief moment for user to see message
                    # Streamlit reruns on state change, so we might need a small pause or explicit rerun here
                    # For simplicity, we'll let the ingestion spinner cover the transition


        with st.spinner(f"Ingesting {ingest_type}... Please wait."):
            if ingest_type in ["PDF", "CSV"]: # Updated to include CSV
                files = {"file": (data_or_file.name, data_or_file.getvalue(), f"application/{ingest_type.lower()}")}
                result = call_fastapi(endpoint, files=files)
            else: # URL
                result = call_fastapi(endpoint, data=data_or_file)

            if result.get("status") == "error":
                st.error(f"{ingest_type} Ingestion Failed: {result['message']}")
            else:
                st.success(f"{success_message}: {result.get('message', 'Success')}")
                if ingestion_mode == "Clear existing & ingest new":
                    # If we cleared, we start a fresh chat as well.
                    st.session_state.messages = []
                    st.session_state.messages.append({"role": "assistant", "content": f"Knowledge base updated with {ingest_type} content. Ask me anything about it!"})

                # Force rerun to clear chat history and update UI if mode is "Clear existing & ingest new"
                st.rerun()

    # PDF Ingestion
    st.subheader("Upload PDF")
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_uploader")
    if uploaded_pdf and st.button("Ingest PDF", key="ingest_pdf_btn"):
        handle_ingestion("PDF", uploaded_pdf, "/ingest/pdf", "PDF Ingested")

    # NEW: CSV Ingestion
    st.subheader("Upload CSV")
    uploaded_csv = st.file_uploader("Upload a CSV file", type=["csv"], key="csv_uploader")
    if uploaded_csv and st.button("Ingest CSV", key="ingest_csv_btn"):
        handle_ingestion("CSV", uploaded_csv, "/ingest/csv", "CSV Ingested")

    # URL Ingestion
    st.subheader("Ingest Web Link (URL)")
    url_to_ingest = st.text_input("Enter a URL to ingest", key="url_input")
    if st.button("Ingest URL", key="ingest_url_btn"):
        if url_to_ingest:
            handle_ingestion("URL", {"url": url_to_ingest}, "/ingest/url", "URL Ingested")
        else:
            st.warning("Please enter a URL.")

# --- Chat Interface Section ---
st.header("Ask Your Ingested Knowledge")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Call FastAPI /ask endpoint
            result = call_fastapi("/ask", data={"question": prompt})

            if result.get("status") == "error":
                response = f"Error: {result['message']}"
                st.error(response)
            else:
                response = result.get("answer", "No answer found.")
                st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

# Button to clear chat history (doesn't affect knowledge base)
if st.button("Clear Chat History", key="clear_chat_history_btn"):
    st.session_state.messages = []
    st.rerun()
