import streamlit as st
from typing import List
from pathlib import Path
from qa_tool import QATool 

# Initialize the QATool instance
qa_tool = QATool()


# Sidebar for document uploads
st.sidebar.title("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload document files (PDF or DOCX)", 
    accept_multiple_files=True, 
    type=["pdf", "docx"]
)

# Custom CSS
st.markdown("""
<style>
            
/* Override Streamlit's default input border */
[data-testid="stTextInputRootElement"] {
    border-color: transparent !important;
}

/* Also remove focus border */
[data-testid="stTextInputRootElement"]:focus-within {
    border-color: transparent !important;
    box-shadow: none !important;
}

/* Remove hover effect */
[data-testid="stTextInputRootElement"]:hover {
    border-color: transparent !important;
}
/* Hide tooltip and help text in all states */
[data-testid="stTextInputRootElement"]::after,
[data-testid="stTextInputRootElement"]:hover::after,
[data-testid="stTextInputRootElement"]:focus::after,
[data-testid="stTextInputRootElement"]:active::after {
    display: none !important;
    content: none !important;
    opacity: 0 !important;
    visibility: hidden !important;
}

/* Hide all tooltip containers and help text elements */
.st-emotion-cache-1gulkj5,
.st-emotion-cache-1adg2ll,
.st-emotion-cache-1ako809,
.st-emotion-cache-1ph64tk,
.st-emotion-cache-16idsys,
.st-emotion-cache-1qg05tj {
    display: none !important;
    opacity: 0 !important;
    visibility: hidden !important;
}

/* Disable pointer events on tooltip containers */
[data-baseweb="tooltip"],
[role="tooltip"] {
    display: none !important;
    pointer-events: none !important;
}
* Global Font Style */
body {
    font-family: 'Arial', sans-serif;
    color: #333;
}

/* Title Typing Animation with Cursor Fully Transparent */
@keyframes typing {
    from { width: 0; }
    to { width: 100%; }
}
@keyframes blink {
    0%, 100% { border-color: transparent; } /* Fully transparent cursor */
}
.title-animated {
    font-size: 2rem;
    font-family: monospace;
    overflow: hidden;
    white-space: nowrap;
    border-right: 4px solid transparent; /* Transparent cursor */
    width: 100%;
    animation: typing 4s steps(30, end), blink 0.75s step-end infinite;
}
            

/* Layout Adjustments */
.main {
    padding-bottom: 100px;
    max-width: 100% !important;
}
.st-emotion-cache-1y4p8pa {
    max-width: 100% !important;
    padding-left: 5rem !important;
    padding-right: 5rem !important;
}
.chat-container {
    overflow-y: auto;
    padding: 20px;
    box-sizing: border-box;
}
/* General Container Adjustments */
.input-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 10px;
    z-index: 1000;
    display: flex;
    align-items: center;
    gap: 10px; /* Add space between input and button */
    border: none; /* Remove border */
    outline: none; /* Remove focus outline */
}
.stTextArea textarea {
    min-height: 50px !important;
    max-height: 200px !important;
    resize: vertical !important;
    overflow-y: auto !important;
    padding: 12px !important;
    border-radius: 15px !important;
    border: 1px solid #ddd !important;
    font-size: 16px !important;
    line-height: 1.5 !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}
.stTextArea textarea:focus {
    border-color: #4A4E69 !important;
    box-shadow: 0 0 0 1px #4A4E69 !important;
}
/* Text Input Styling */
.stTextInput input {
    flex: 1; /* Input takes up remaining width */
    padding: 12px 15px !important;
    font-size: 16px !important;
    border: 1px solid transparent; /* Remove visible border */
    border-radius: 8px;
    background-color: #333; /* Dark theme */
    color: #fff; /* Light text for contrast */
}
.stTextInput input:hover,
.stTextInput input:focus {
    border-color: #9A8C98 !important; /* Red border when focused */
    outline: none !important;
    box-shadow: none !important;
}

/* Button Styling */
button[kind="primary"] {
    background-color: #4A4E69; /* Red background for button */
    color: #fff; /* White text */
    font-size: 16px;
    padding: 12px 20px;
    border: none; /* No border */
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
button[kind="primary"]:hover {
    background-color: #4A4E69; /* Darker red on hover */
}
button[kind="primary"]:focus {
    outline: none; /* Remove focus outline */
    box-shadow: none; /* Remove focus shadow */
}

/* Fix Streamlit Submit Button Alignment */
[data-testid="stFormSubmitButton"] {
    margin: 0;
    padding: 0;
}
/* Message Styling */
.message-bubble {
    font-size: 16px;
    padding: 10px;
    border-radius: 10px;
    max-width: 100%;
    word-wrap: break-word;
    margin: 5px 0;
    color: #000; /* Set message text color to black */
}
.chat-row {
    display: flex;
    flex-direction: row;
    margin: 5px;
    width: 100%;
}
.message-container {
    border-radius: 10px;
    padding: 10px;
    margin: 5px;
    max-width: 100%;
}
/* User Message Styling */
.user-message .message-bubble {
    background-color: #4A4E69; 
    margin-left: auto;
    color: #fff; /* White text for user messages */
}

/* Bot Message Styling */
.bot-message .message-bubble {
    background-color: #f1f0f0; /* Light gray background for bot */
    margin-right: auto;
}
/* Typing indicator */
.typing-indicator {
    background-color: #f1f0f0;
    border-radius: 15px;
    padding: 15px;
    display: inline-block;
    margin-left: 10px;
    position: relative;
}

.dot {
    display: inline-block;
    width: 3px;
    height: 3px;
    margin: 0 1px;
    background-color: black;
    border-radius: 50%;
    animation: bounce 1.4s infinite;
}

.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-8px); }
}
</style>
""", unsafe_allow_html=True)

# Animated title
st.markdown('<div class="title-animated">Hello, How can I assist you today ?</div>', unsafe_allow_html=True)


# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False

# Main container
st.markdown('<div class="main">', unsafe_allow_html=True)

# Chat messages container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
# Update message display logic
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"""
            <div class='message-container user-message'>
                <div class='message-bubble'>
                    <div class='user-icon'>👤</div>
                    {chat['message']}
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='message-container bot-message'>
                <div class='message-bubble'>
                    <div class='bot-icon'>🤖</div>
                    {chat['message']}
                </div>
            </div>
        """, unsafe_allow_html=True)

# Show typing indicator
# Add typing indicator when waiting for response
if st.session_state.waiting_for_response:
    st.markdown("""
        <div class='message-container bot-message'>
            <div class='message-bubble typing-indicator'>
                <div class='bot-icon'>🤖</div>
                <div class='dot'></div>
                <div class='dot'></div>
                <div class='dot'></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input container
st.container().markdown('<div class="input-container">', unsafe_allow_html=True)
with st.form(key='message_form', clear_on_submit=True):
    col1, col2 = st.columns([5, 1])  # Adjust column ratios for input and button
    with col1:
        question = st.text_input(
            "",
            key="user_input",
            placeholder="Ask your Question here...",
            label_visibility="collapsed"
        )
    with col2:
        submit_button = st.form_submit_button("Send")
    
    if submit_button and question.strip():
        st.session_state.chat_history.append({"role": "user", "message": question})
        st.session_state.waiting_for_response = True
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Process the response after rerun
if st.session_state.waiting_for_response:
    document_paths = []
    if uploaded_files:
        for file in uploaded_files:
            file_path = Path(f"temp_files/{file.name}")
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "wb") as f:
                file_content = file.getvalue()
                f.write(file_content)
            if file_path.exists():
                document_paths.append(str(file_path.absolute()))
            else:
                st.error(f"Failed to save file: {file.name}")
    
    with st.spinner(text=''):
        answer = qa_tool.get_answer(st.session_state.chat_history[-1]["message"], document_paths)
        answer = answer.replace("Answer:", "", 1).strip()
    
    st.session_state.chat_history.append({"role": "bot", "message": answer})
    st.session_state.waiting_for_response = False
    st.rerun()
    st.stop()
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️")