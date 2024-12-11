from typing import List, Dict, Any
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import wikipedia
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.tools import StructuredTool, tool
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from pydantic import BaseModel, Field
from difflib import SequenceMatcher
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import requests
import os

load_dotenv()

class QATool:
    class SearchDocumentInput(BaseModel):
        query: str = Field(description="The search query")
        file_paths: List[str] = Field(description="A list of paths to document files")
    
    class TavilySearchInput(BaseModel):
        query: str = Field(description="The search query for Tavily")
        search_depth: str = Field(default="basic", description="Depth of the search: 'basic' or 'advanced'")
        topic: str = Field(default="general", description="Category of the search: 'general' or 'news'")
        days: int = Field(default=None, description="Number of days back to search (for 'news' topic)")
        max_results: int = Field(default=5, description="Maximum number of results to return")
        include_answer: bool = Field(default=True, description="Include a short answer in the response")

    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        self.llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-002", temperature=0)
        self.search_wikipedia_tool = StructuredTool.from_function(self.search_wikipedia)
        self.agent = self._initialize_agent()

    def _initialize_agent(self):
        search_wikipedia_tool = StructuredTool.from_function(self.search_wikipedia)
        search_document_tool = self.search_document_tool
        tavily_web_search_tool = self.tavily_web_search_tool
        system_message = '''Respond as helpfully and accurately as possible. You have access to the following tools:
        {tools}
        Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
        Valid "action" values: "Final Answer" or {tool_names}
        Provide only ONE action per $JSON_BLOB, as shown:
        {{
          "action": $TOOL_NAME,
          "action_input": $INPUT
        }}
        Follow this format:
        Question: input question to answer
        Thought: consider previous and subsequent steps
        Action:
        {{
          $JSON_BLOB
        }}
        Observation: action result
        ... (repeat Thought/Action/Observation N times)
        Thought: I know what to respond
        Action:
        {{
          "action": "Final Answer",
          "action_input": "Final response to human"
        }}
        Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation'''

        human_message = '''
        {input_data}
        {question}
        {agent_scratchpad}
        (reminder to respond in a JSON blob no matter what)'''

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("human", human_message),
        ])
        
        return create_structured_chat_agent(
            llm=self.llm,
            tools=[search_wikipedia_tool, search_document_tool, tavily_web_search_tool],
            prompt=prompt,
        )

    def search_wikipedia(self, query: str) -> str:
        """Searches Wikipedia for the given query and returns a summary of the top 3 results."""
        page_titles = wikipedia.search(query)
        summaries = []
        for page_title in page_titles[:3]:
            try:
                wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
                summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
            except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
                pass
        return "\n\n".join(summaries) if summaries else "No good Wikipedia Search Result was found"

    @staticmethod
    def load_document(file_path: str) -> List[Any]:
        """Loads a document from the specified file path."""
        if file_path.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported document type")
        return loader.load()

    @staticmethod
    def keyword_overlap( query: str, candidate: str) -> float:
        """Calculates the keyword overlap ratio between query and candidate."""
        return SequenceMatcher(None, query, candidate).ratio()
    @staticmethod
    def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> list:
        """Split text into chunks with a specified overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    @staticmethod
    @tool(args_schema=SearchDocumentInput)
    def search_document_tool(query: str, file_paths: list) -> str:
        """Search for the query across multiple documents using a hybrid approach with aggregation and summarization."""
        try:
            all_documents = []
            # Load and combine documents from all file paths
            for file_path in file_paths:
                documents = QATool.load_document(file_path)
                all_documents.extend(documents)
            
            chunk_size = 1000
            overlap = 200
            # Process texts as chunks with overlap
            chunks = []
            for document in all_documents:
                text = document.page_content
                chunks.extend(QATool.split_text_into_chunks(text, chunk_size, overlap))
            # Create dense embeddings with SentenceTransformer
            embedding_model = SentenceTransformer("all-mpnet-base-v2")
            dense_embeddings = embedding_model.encode(chunks)
            query_embedding = embedding_model.encode(query)

            # Perform dense similarity search
            dense_results = [
                (chunk, 1 - cosine(query_embedding, doc_embedding))
                for chunk, doc_embedding in zip(chunks, dense_embeddings)
            ]
            dense_results = sorted(dense_results, key=lambda x: -x[1])
            #print("Dense Result", dense_results)

            # Sparse retrieval using BM25
            bm25 = BM25Okapi([chunk.split() for chunk in chunks])
            sparse_scores = bm25.get_scores(query.split())
            sparse_results = sorted(
                zip(all_documents, sparse_scores), key=lambda x: x[1], reverse=True
            )
            #print("\n")
            #print("Sparse Result", sparse_results)
            # Combine dense and sparse results
            hybrid_results = []
            for dense_result in dense_results:
                dense_doc, dense_score = dense_result
                sparse_score = next(
                    (score for chunk, score in sparse_results if chunk == dense_doc),
                    0,
                )
                keyword_score = QATool.keyword_overlap(query, dense_doc)
                hybrid_results.append((dense_doc, dense_score, sparse_score, keyword_score))
        
                
            # Weighted scoring
            hybrid_results = [
                (doc, 0.7 * dense_score + 0.2 * sparse_score + 0.1 * keyword_score)
                for doc, dense_score, sparse_score, keyword_score in hybrid_results
            ]
            #print("\n")
            #print("Hybrid Result", hybrid_results)
            # Sort hybrid results by weighted score
            sorted_results = sorted(hybrid_results, key=lambda x: -x[1])
            #print("\n")
            # Aggregate relevant chunks
            aggregated_content = ""
            threshold = 0.3  # Aggregation threshold
            for chunk, score in sorted_results:
                if abs(score) <= threshold:
                    aggregated_content += chunk+ "\n\n"

            # Validate and summarize aggregated content
            if aggregated_content.strip():
                if QATool.validate_result(query, aggregated_content):
                    # Summarize aggregated content
                    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-002", temperature=0.7)
                    summary = llm.invoke(f"Summarize this content: {aggregated_content} with a special focus on the answer to this question:  {query}.")
                    return summary

            return "No relevant information found in the documents."
        except Exception as e:
            return f"Error during document search: {str(e)}"
    
    @staticmethod
    def validate_result(query: str, candidate: str) -> bool:
      """Validates the relevance of a candidate answer to a query."""
      llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-002", temperature=1)
      prompt = f"Query: {query}\nCandidate Result: {candidate}\nDoes the candidate provide a relevant answer to the query? Respond with 'Yes' or 'No' and explain briefly."
      validation_response = llm.invoke(prompt)
      return "yes" in validation_response.content.lower()
    
    @staticmethod
    @tool(args_schema=TavilySearchInput)
    def tavily_web_search_tool(
        query: str, 
        search_depth: str = "basic", 
        topic: str = "general", 
        days: int = None, 
        max_results: int = 5, 
        include_answer: bool = True,
        include_images: bool = False,
        include_image_descriptions: bool = False,
        include_raw_content: bool = False,
        include_domains: List[str] = None,
        exclude_domains: List[str] = None,
    ) -> Dict[str, Any]:
        """Search the web using Tavily."""
        try:
            # Retrieve the API key
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise ValueError("TAVILY_API_KEY is missing or not set in the environment.")

            # Prepare the request payload
            payload = {
                "api_key": api_key,
                "query": query,
                "search_depth": search_depth,
                "topic": topic,
                "max_results": max_results,
                "include_answer": include_answer,
                "include_images": include_images,
                "include_image_descriptions": include_image_descriptions,
                "include_raw_content": include_raw_content,
                "include_domains": include_domains or [],
                "exclude_domains": exclude_domains or [],
            }

            # Only add "days" if it's provided (not None)
            if days is not None:
                payload["days"] = days

            # Send POST request to Tavily
            url = "https://api.tavily.com/search"
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx, 5xx)
            return response.json()  # Return the Tavily response directly
        
        except ValueError as ve:
            return {"error": str(ve)}  # Return error if API key is missing
        except requests.exceptions.RequestException as re:
            return {"error": str(re)}  # Return request-related errors
    
    def get_answer(self, question: str, document_paths: list = None):
      """Gets an answer to a question, optionally using a document."""
      try:
          # Ensure memory exists
          if not hasattr(self, "memory"):
              from langchain.memory import ConversationBufferMemory
              self.memory = ConversationBufferMemory(return_messages=True)

          # Convert chat history into a list of messages
          chat_history = self.memory.chat_memory.messages if self.memory.chat_memory else []

          if document_paths:
              input_data = {"query": question, "file_paths": document_paths}
              print(f"Input Data: {input_data}")  # Debugging
              formatted_prompt = {
                  "question": question,
                  "input_data": input_data,
                  "chat_history": chat_history,
                  "agent_scratchpad": "",
              }
              #print(f"Formatted Prompt: {formatted_prompt}")  # Debugging

              agent_executor = AgentExecutor(agent=self.agent, tools=[self.search_document_tool, self.search_wikipedia_tool, self.tavily_web_search_tool], handle_parsing_errors=True, verbose=True)
              response = agent_executor.invoke(formatted_prompt)
              
              if "No relevant information found" in response["output"]:
                  formatted_prompt = {
                      "question": question,
                      "input_data": {},  # No document context
                      "chat_history": chat_history,
                      "agent_scratchpad": "",
                  }
                  agent_executor = AgentExecutor(agent=self.agent, tools=[self.search_wikipedia_tool, self.tavily_web_search_tool], handle_parsing_errors=True, verbose=True)
                  response = agent_executor.invoke(formatted_prompt)
          else:
              formatted_prompt = {
                  "question": question,
                  "input_data": {},  # No document context
                  "chat_history": chat_history,
                  "agent_scratchpad": "",
              }
              #print(f"Formatted Prompt: {formatted_prompt}")  # Debugging

              agent_executor = AgentExecutor(agent=self.agent, tools=[self.search_wikipedia_tool, self.tavily_web_search_tool], handle_parsing_errors=True)
              response = agent_executor.invoke(formatted_prompt)

          # Save the conversation to memory
          self.memory.save_context(
              inputs={"question": question},
              outputs={"output": response["output"]},
          )

          return f'Answer: {response["output"]}'
      except Exception as e:
          return f"Error during QA process: {str(e)}"

"""
# Usage Example
tool = QATool()
question = "Hello, can you tell me the name of the techniques implemented in this report?"
document_paths = [
    "/Users/zeinebtekaya/Desktop/IIA 5/Rapport_Stage/RapportFinal.pdf"
]
answer = tool.get_answer(question, document_paths)
print(answer)
"""