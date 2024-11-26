import hmac
from typing import List

import openai
import pinecone
import streamlit as st
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=st.secrets["PERSONAL_PINECONE_KEY"])
index = pc.Index("consilience-google-drive")

# Initialize Mongo
mongo_pwd = st.secrets["MONGO_TEST_DB_PASSWORD"]
uri = f"mongodb+srv://adriel:{mongo_pwd}@cluster0.9nejc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)

MONGO = client["test"]["consilience-google-drive"]

# Initialize Sentence Embedder
EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Set OpenAI API key
openai.api_key = st.secrets["PERSONAL_OPENAI_KEY"]

# Hide the Streamlit buttons
hide_streamlit_styles1 = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_styles1, unsafe_allow_html=True)

hide_streamlit_styles2 = """
    <style>
    ._profileContainer_gzau3_53 {
        display: none;
    }
    ._link_gzau3_10 {
        display: none;
    }
    </style>
"""

st.markdown(hide_streamlit_styles2, unsafe_allow_html=True)

hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def query_mongo(ids: List[str]) -> List[dict]:
    """
    Query MongoDB for documents and return them in the same order as the input IDs.

    Args:
        ids (List[str]): List of document IDs to query

    Returns:
        List: List of documents in the same order as input IDs
    """
    # Get all matching documents
    query = {"_id": {"$in": ids}}
    raw_docs = list(MONGO.find(query))

    # Return documents in the same order as input ids
    docs_map = {str(doc["_id"]): doc for doc in raw_docs}
    ordered_docs = [docs_map[id_] for id_ in ids if id_ in docs_map]

    return ordered_docs


def get_embedding(text) -> List[float]:
    return EMBEDDING_MODEL.encode(text).tolist()


def query_pinecone(embedding, top_k=10) -> List[dict]:
    query_result = index.query(vector=embedding, top_k=top_k, include_metadata=False)

    return query_result['matches']


def generate_context(docs: List[dict]) -> str:
    context = "\n"
    for d in docs:
        context += f"File Name: {d['file_name']}\n"
        context += f"Text: {d['text']}\n"

    return context


# Function to generate a response using OpenAI GPT, injecting the retrieved context
def generate_response(question: str, context: str) -> str:
    prompt = f"Question: {question}\n\nRelevant Information:\n{context}\n\nAnswer:"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """
            You are a world-class insurance agent. Rely only on the context to generate an answer,and your own internal
            insurance regulation understanding. You can summarize or expand on the context but don't inject any new
            information. If there is a regulation number provided in the context that is relevant, cite it. If the
            context provided isn't helpful or is not relevant to the question just say 'I don't know'.
            """
             },
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0
    )

    return response.choices[0].message.content.strip()


def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Streamlit app layout
st.title("Insurance GPT")

user_question = st.text_input("Ask a question:")

if user_question:
    with st.spinner("Processing..."):
        # Step 1: Get embedding for the user question
        question_embedding = get_embedding(user_question)

        # Step 2: Query Pinecone for the relevant document IDs
        doc_ids = [doc["id"] for doc in query_pinecone(question_embedding, top_k=10)]

        # Step 3: Query Mongo for the documents themselves
        docs = query_mongo(doc_ids)

        # Step 4: Inject the relevant documents into the context for response generation
        if docs:
            context = generate_context(docs)
            response = generate_response(user_question, context)

            # Display the response
            # Display the response as plain text with larger font
            st.markdown("<h2>Answer:</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 22px;'>{response}</p>", unsafe_allow_html=True)
            st.markdown("\n\n\n\n\n\n")

            # Optionally, display the relevant documents
            st.write("\n\n\n**Relevant Information:**")
            for i, doc in enumerate(docs):
                # Markdown requires two newline characters to add a line break:
                text = doc['text'].replace('\n', '\n\n')
                st.markdown(
                    f"<p style='font-size: 15px;'><b>From {doc['file_name']}:</b></p>"
                    f"<p style='font-size: 15px;'>{text}</p>",
                    unsafe_allow_html=True
                )
        else:
            st.write("No relevant documents found.")
