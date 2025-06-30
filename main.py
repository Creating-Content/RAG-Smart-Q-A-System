# smart_qa_system/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import shutil
import gc # Required for garbage collection
import asyncio # Required for async sleep
from contextlib import asynccontextmanager # New import for lifespan events
from pydantic import BaseModel # For request body models

# LangChain specific imports
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader # Added WebBaseLoader
from langchain_community.document_loaders import CSVLoader # NEW: For CSV ingestion
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")

CHROMA_DB_PATH = "./my_new_chroma_db"
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-flash"

# --- Global RAG Components ---
vector_store = None
conversation_chain = None
memory = None
embeddings = None
llm = None

# --- Prompt Template for Conversational Retrieval Chain ---
# MODIFIED PROMPT HERE TO BETTER HANDLE SUMMARIZATION
qa_prompt = PromptTemplate(
    template="""You are a helpful assistant that answers questions based on the provided context.
    If the question asks for a summary, synthesize the information from the context to provide a concise summary.
    If the question asks for a direct answer, provide it based on the context.
    If you cannot find the answer or sufficient information to summarize within the provided context,
    please state: "I am unable to find sufficient information in the provided documents to answer or summarize."
    Do not make up answers.

    Context: {context}
    Question: {question}
    Answer:""",
    input_variables=["context", "question"]
)

# --- FastAPI Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI.
    Initializes LangChain components (LLM, Embeddings, Memory) on application startup.
    Yields control to the application, and then runs shutdown code when the app stops.
    """
    global vector_store, conversation_chain, memory, embeddings, llm

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.5)

        try:
            if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
                vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
                print("Loaded existing ChromaDB.")
            else:
                print("ChromaDB directory is empty or does not exist. It will be initialized on first ingestion.")
        except Exception as db_e:
            print(f"Could not load existing ChromaDB (may not exist yet): {db_e}")
            vector_store = None

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Initialize conversation_chain if vector_store was loaded on startup
        if vector_store:
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(),
                memory=memory,
                combine_docs_chain_kwargs={"prompt": qa_prompt}
            )
            print("Conversational chain initialized with existing ChromaDB.")

        print("FastAPI app started. RAG components (LLM, Embeddings, Memory) ready. Vector store initialized/loaded.")
        yield # This is where FastAPI app starts receiving requests
        # --- Any shutdown code would go here after `yield` ---
        print("FastAPI app shutting down.")

    except Exception as e:
        print(f"Error during RAG component initialization: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to initialize RAG components: {e}")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Smart Document Q&A System",
    description="Ask questions about your documents, web links, and YouTube videos.", # Description updated below
    version="0.1.0",
    lifespan=lifespan # Assign the lifespan handler
)

# Update description to reflect CSV support
app.description = "Ask questions about your documents (PDF, CSV), web links."

origins = [
    "http://localhost",
    "http://localhost:8501", # Streamlit's default port
    # Add any other origins if you deploy your Streamlit app elsewhere
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)# --- Endpoints ---

@app.get("/")
async def read_root():
    """
    Root endpoint for the Smart Q&A System.
    """
    return {"message": "Welcome to the Smart Document Q&A System API! Visit /docs for API documentation."}

@app.get("/health")
async def health_check():
    """
    Health check endpoint to ensure the API is running.
    """
    return {"status": "ok", "message": "API is healthy"}

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Ingest a PDF file into the knowledge base for Q&A.
    """
    global vector_store, conversation_chain, memory, embeddings, llm

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported.")

    if embeddings is None or llm is None:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RAG components not initialized. Server startup failed or misconfigured.")

    # Reset memory for new document ingestion
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    try:
        # 1. Save the uploaded file temporarily
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # 2. Load the PDF document
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No content found in PDF.")

        # 3. Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # 4. Generate embeddings and add to vector store
        if vector_store is None:
            vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)
            vector_store.persist()
            print(f"Initialized ChromaDB with {len(chunks)} chunks from {file.filename}")
        else:
            vector_store.add_documents(chunks)
            vector_store.persist()
            print(f"Added {len(chunks)} chunks from {file.filename} to existing ChromaDB.")

        # 5. Re-initialize the conversational chain after vector_store is populated
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": f"Successfully ingested {len(chunks)} chunks from {file.filename}."}
        )

    except Exception as e:
        print(f"Error during PDF ingestion: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to ingest PDF: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# Pydantic model for URL ingestion
class URLIngestRequest(BaseModel):
    url: str

@app.post("/ingest/url")
async def ingest_url(request: URLIngestRequest):
    """
    Ingest content from a web URL into the knowledge base.
    """
    global vector_store, conversation_chain, memory, embeddings, llm

    if embeddings is None or llm is None:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RAG components not initialized. Server startup failed or misconfigured.")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    try:
        # Validate URL (basic check)
        if not (request.url.startswith("http://") or request.url.startswith("https://")):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid URL format. Must start with http:// or https://")

        # Load content from URL
        loader = WebBaseLoader(request.url)
        documents = loader.load()

        if not documents:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No content found at the provided URL or failed to load.")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Add to vector store
        if vector_store is None:
            vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)
            vector_store.persist()
            print(f"Initialized ChromaDB with {len(chunks)} chunks from {request.url}")
        else:
            vector_store.add_documents(chunks)
            vector_store.persist()
            print(f"Added {len(chunks)} chunks from {request.url} to existing ChromaDB.")

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": f"Successfully ingested {len(chunks)} chunks from {request.url}."}
        )

    except Exception as e:
        print(f"Error during URL ingestion: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to ingest URL: {e}")

# NEW: CSV Ingestion Endpoint
@app.post("/ingest/csv")
async def ingest_csv(file: UploadFile = File(...)):
    """
    Ingest a CSV file into the knowledge base for Q&A.
    """
    global vector_store, conversation_chain, memory, embeddings, llm

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only CSV files are supported.")

    if embeddings is None or llm is None:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RAG components not initialized. Server startup failed or misconfigured.")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    try:
        # 1. Save the uploaded file temporarily
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # 2. Load the CSV document
        # CSVLoader can take a file_path directly
        loader = CSVLoader(file_path)
        documents = loader.load()

        if not documents:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No content found in CSV.")

        # 3. Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # 4. Generate embeddings and add to vector store
        if vector_store is None:
            vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)
            vector_store.persist()
            print(f"Initialized ChromaDB with {len(chunks)} chunks from {file.filename}")
        else:
            vector_store.add_documents(chunks)
            vector_store.persist()
            print(f"Added {len(chunks)} chunks from {file.filename} to existing ChromaDB.")

        # 5. Re-initialize the conversational chain after vector_store is populated
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": f"Successfully ingested {len(chunks)} chunks from {file.filename}."}
        )

    except Exception as e:
        print(f"Error during CSV ingestion: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to ingest CSV: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# Pydantic model for the Q&A request body
class QuestionRequest(BaseModel):
    question: str

## Clear Knowledge Base Endpoint

@app.post("/clear_db")
async def clear_database():
    """
    Clears the entire ChromaDB knowledge base and resets the conversation chain.
    Attempts to explicitly release file locks before deleting the directory.
    """
    global vector_store, conversation_chain, memory, embeddings, llm # Ensure all global variables are listed

    try:
        # Step 1: Force release of the vector_store object (and its file handle)
        if vector_store is not None:
            print("Attempting to release vector_store object and its file handles...")
            vector_store = None # Remove reference to the Chroma object
            gc.collect() # Explicitly request Python to run garbage collection
            await asyncio.sleep(0.2) # Give a small asynchronous pause for the OS to release the handle

        # Step 2: Attempt to remove the ChromaDB directory
        if os.path.exists(CHROMA_DB_PATH):
            print(f"Attempting to delete ChromaDB directory: {CHROMA_DB_PATH}")
            shutil.rmtree(CHROMA_DB_PATH)
            print(f"ChromaDB at {CHROMA_DB_PATH} cleared successfully.")
        else:
            print(f"ChromaDB directory '{CHROMA_DB_PATH}' does not exist, nothing to clear.")

        # Step 3: Re-initialize all global RAG components to a fresh state
        # This ensures that after clearing, the system is ready for new ingestion
        # and has no stale references.
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.5)

        vector_store = None # Ensure it's None; it will be created on the next ingestion
        conversation_chain = None # The chain needs to be rebuilt once vector_store is populated
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) # Start with fresh memory

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Knowledge base cleared successfully. Ready for new ingestion."}
        )
    except Exception as e:
        print(f"Error during clear database: {e}")
        # Provide a more specific error message for file locking issues
        if "[WinError 32]" in str(e) or "The process cannot access the file" in str(e):
            detail_msg = f"Failed to clear database: The file is still locked. This often happens if the server didn't fully release the previous connection. Please ensure all Python processes related to this project are terminated (check Task Manager) and try again. Error: {e}"
        else:
            detail_msg = f"Failed to clear database: {e}"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail_msg)


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the ingested documents.
    """
    global conversation_chain

    if conversation_chain is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No documents have been ingested yet. Please ingest a PDF, URL, or CSV file first.")

    try:
        response = conversation_chain.invoke({"question": request.question})
        answer = response.get("answer", "No answer found.")

        # Optional: Retrieve and return source documents (uncomment if desired)
        # retriever = vector_store.as_retriever()
        # retrieved_docs = retriever.get_relevant_documents(request.question)
        # sources = [{"page_content": doc.page_content, "source": doc.metadata.get('source', 'N/A'), "page": doc.metadata.get('page', 'N/A')} for doc in retrieved_docs]

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "question": request.question,
                "answer": answer,
                # "sources": sources # Uncomment if you want to return sources
            }
        )
    except Exception as e:
        print(f"Error during Q&A: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to answer question: {e}")
