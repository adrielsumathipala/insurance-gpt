from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, Security, Depends, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import pinecone
from state_to_files import STATE_TO_FILE_MAP
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import os
import logging
from tiktoken import encoding_for_model
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GPT_MODEL = "gpt-3.5-turbo"
# Initialize tokenizer for gpt-3.5-turbo
ENCODING = encoding_for_model(GPT_MODEL)
MAX_TOKENS = 16385  # max context length for gpt-3.5-turbo

# Disabling to avoid potential deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Environment variables class
class EnvVars:
    PERSONAL_PINECONE_KEY = os.environ["PERSONAL_PINECONE_KEY"]
    MONGO_TEST_DB_PASSWORD = os.environ["MONGO_TEST_DB_PASSWORD"] 
    PERSONAL_OPENAI_KEY = os.environ["PERSONAL_OPENAI_KEY"]
    INSURANCE_GPT_API_KEY = os.environ["INSURANCE_GPT_API_KEY"]
    APP_PASSWORD = os.environ.get("APP_PASSWORD", "lula")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(debug=True)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    # TODO: Remove "*", only while testing deployment on EB
    allow_origins=["http://localhost:8000", "https://theinsurancegpt.com", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=EnvVars.PERSONAL_PINECONE_KEY)
index = pc.Index("consilience-google-drive")

# Initialize Mongo
mongo_pwd = EnvVars.MONGO_TEST_DB_PASSWORD
uri = f"mongodb+srv://adriel:{mongo_pwd}@cluster0.9nejc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)
MONGO = client["test"]["consilience-google-drive"]

# Initialize Sentence Embedder
EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Set OpenAI API key
openai.api_key = EnvVars.PERSONAL_OPENAI_KEY

# API Key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == EnvVars.INSURANCE_GPT_API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=403, detail="Could not validate API key"
    )

class QuestionRequest(BaseModel):
    question: str
    state: str

def query_mongo(ids: List[str]) -> List[dict]:
    """Query MongoDB for documents and return them in order."""
    query = {"_id": {"$in": ids}}
    raw_docs = list(MONGO.find(query))
    docs_map = {str(doc["_id"]): doc for doc in raw_docs}
    ordered_docs = [docs_map[id_] for id_ in ids if id_ in docs_map]
    return ordered_docs

def get_embedding(text) -> List[float]:
    return EMBEDDING_MODEL.encode(text).tolist()


def generate_filter_condition(state: str) -> dict:
    fnames = STATE_TO_FILE_MAP[state]

    return {
        "file_name" : {"$in": fnames}
    }

def query_pinecone(embedding: List[float], filter_condition: dict={}, top_k: int=10) -> List[dict]:
    query_result = index.query(vector=embedding, top_k=top_k, filter=filter_condition,include_metadata=False)
    
    return query_result['matches']

def generate_context(docs: List[dict]) -> str:
    context = "\n"
    for d in docs:
        context += f"File Name: {d['file_name']}\n"
        context += f"Text: {d['text']}\n"
    return context

def generate_response(question: str, context: str) -> str:
    SYSTEM_PROMPT = """
    You are a world-class insurance agent. Rely only on the context to generate an answer,and your own internal
    insurance regulation understanding. You can summarize or expand on the context but don't inject any new
    information. If there is a regulation number provided in the context that is relevant, cite it. If the
    context provided isn't helpful or is not relevant to the question just say 'I don't know'.
    """
    
    # Calculate tokens for system message and question
    system_tokens = len(ENCODING.encode(SYSTEM_PROMPT))
    question_tokens = len(ENCODING.encode(f"Question: {question}\n\nRelevant Information:\n\n\nAnswer:"))
    
    # Calculate available tokens for context
    available_tokens = MAX_TOKENS - system_tokens - question_tokens - 1200 # Add buffer for response

    # Truncate context if needed
    context_tokens = ENCODING.encode(context)
    if len(context_tokens) > available_tokens:
        context = ENCODING.decode(context_tokens[:available_tokens])
        context += "..."

    prompt = f"Question: {question}\n\nRelevant Information:\n{context}\n\nAnswer:"
    response = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0
    )

    return response.choices[0].message.content.strip()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    states = list(STATE_TO_FILE_MAP.keys())
    return templates.TemplateResponse("index.html", {
        "request": request,
        "api_key": EnvVars.INSURANCE_GPT_API_KEY,
        "states": states
    })

@app.post("/ask")
async def ask_question(
    question_request: QuestionRequest,
    api_key: APIKey = Depends(get_api_key)
):
    try:
        logger.info(question_request)
        # Get embedding for the user question
        question_embedding = get_embedding(question_request.question)

        # Query Pinecone for relevant document IDs
        filter_condition = generate_filter_condition(question_request.state)
        doc_ids = [doc["id"] for doc in query_pinecone(question_embedding, filter_condition, top_k=10)]

        # Query Mongo for the documents
        docs = query_mongo(doc_ids)

        if docs:
            context = generate_context(docs)
            response = generate_response(question_request.question, context)
            
            return {
                "answer": response,
                "documents": [{"file_name": doc["file_name"], "text": doc["text"]} for doc in docs]
            }
        else:
            return {
                "answer": "No relevant documents found.",
                "documents": []
            }

    except Exception as e:
        logger.exception("Error in /ask endpoint:")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check_password")
async def check_password(password: str = Form(...)):
    if password == EnvVars.APP_PASSWORD:
        return {"valid": True}
    return {"valid": False}
