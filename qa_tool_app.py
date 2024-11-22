import streamlit as st
from typing import List
from pathlib import Path

from qa_tool import QATool 

# Initialize the QATool instance
qa_tool = QATool()

# App Title
st.title("QA Tool - Document-Assisted Chatbot")

# Sidebar for optional document uploads
st.sidebar.title("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload document files (PDF or DOCX)", 
    accept_multiple_files=True, 
    type=["pdf", "docx"]
)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of dictionaries: {"role": "user" or "bot", "message": str}

# Chatbot Layout
st.subheader("Chat Session")

# Display chat history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['message']}")
    elif chat["role"] == "bot":
        st.markdown(f"**Bot:** {chat['message']}")

# Input area for user question
question = st.text_input("Type your message here:")

# Submit button
if st.button("Send"):
    # Process uploaded documents
    document_paths = []
    if uploaded_files:
        for file in uploaded_files:
            file_path = Path(f"temp_files/{file.name}")
            file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            with open(file_path, "wb") as f:
                f.write(file.read())
            document_paths.append(str(file_path))

    # Add user question to chat history
    if question.strip():
        st.session_state.chat_history.append({"role": "user", "message": question})

        # Get the answer from the QA Tool
        with st.spinner("Processing..."):
            answer = qa_tool.get_answer(question, document_paths)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "bot", "message": answer})
    else:
        st.warning("Please enter a question!")

# Persist chat history and documents
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ by Zeineb Tekaya")
