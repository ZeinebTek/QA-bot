# QA Tool: Document-Assisted Chatbot 🤖📚
This project is a powerful Question-Answering (QA) Tool that combines conversational AI with document-assisted search capabilities. Built using LangChain, Google Generative AI (Gemini), BM25, and Sentence Transformers, this tool can answer questions by intelligently searching through uploaded documents or using web resources like Wikipedia and Tavily.

## Features ✨
- **Document-Assisted QA:** Upload PDF or DOCX files, and the chatbot will extract relevant information to answer your queries.
- **Hybrid Search Techniques:** Combines dense vector embeddings, BM25 sparse retrieval, and keyword overlap for accurate results.
- **Web Search Integration:** Searches Wikipedia or uses Tavily API to retrieve up-to-date answers.
- **Contextual Memory:** Retains conversation history for coherent and contextual responses.
- **Streamlit Interface:** User-friendly web app with interactive chat and document upload options.

## Use Cases 🔍
- **Research Assistance:** Get summarized answers from academic papers or reports.
- **Customer Support:** Upload product manuals or FAQs for an interactive assistant.
- **Knowledge Management:** Answer queries based on company documents or policies.

## Built With 🛠️
- **LangChain:** Conversational framework for LLMs.
- **Google Genertive AI (Gemini):** Advanced LLM for summarization and QA.
- **Rank-BM25:** Sparse retrieval for keyword-based matching.
- **Sentence Transformers:** Dense vector embeddings for semantic search.
- **PyMuPDF & Docx2txt:** Document parsing and loading.
- **Streamlit:** Simple UI for interactive chatbot sessions.

## Getting Started 🚀
1. Clone the repository:
```
git clone https://github.com/<your-username>/qa-tool.git
cd qa-tool
```

2. Create virtual environment and install dependencies:
```
python -m venv your-virtual-env
source your-virtual-env/bin/activate
pip install -r requirements.txt
```

3. Set up environment variables:
- Add your Tavily API Key to .env as TAVILY_API_KEY and Gemini API key as GOOGLE_API_KEY.

4. Run the Streamlit app:
```
streamlit run qa_tool_app.py
```

## Contributions 💡
Contributions, suggestions, and feature requests are welcome! Feel free to open issues or submit pull requests.
## License 📜
This project is open-source and available under the MIT License.

