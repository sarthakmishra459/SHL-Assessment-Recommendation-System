import os
import json
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# Embedding function
def get_embeddings(texts, task_type="retrieval_document", batch_size=100):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        embeddings = [e.values for e in response.embeddings]
        all_embeddings.extend(embeddings)
    return np.array(all_embeddings)

def query_enhancer(query):
    # Placeholder for any query enhancement logic
    prompt = f"""
        You are a query enhancer for SHL assessments.

        Given a vague or general input like: "{query}", convert it into a concise and structured job description.

        Only return a string in the following format:
        [Job Title] [Approximate Test Duration in minutes] [Comma-separated key skills]

        For example:
        "Frontend Developer 40 mins js, html, css, react"

        ⚠️ Do NOT include any explanations, extra words, or formatting like lists or bullet points. Just return the one-line structured response.
    """
    answer = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt)
    return answer.text

# Load or build FAISS index
@st.cache_resource
def load_index_and_metadata():
    if os.path.exists("shl_faiss_index.index") and os.path.exists("shl_metadata.json"):
        index = faiss.read_index("shl_faiss_index.index")
        with open("shl_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        with open("shl_assessments_individual.json", "r", encoding="utf-8") as f:
            individual_data = json.load(f)
        with open("shl_assessments_pre.json", "r", encoding="utf-8") as f:
            prepackaged_data = json.load(f)

        all_data = individual_data + prepackaged_data
        documents = [
            f"{d['name']} {' '.join(d.get('test_types', []))} duration {d.get('duration', '')} minutes"
            for d in all_data
        ]

        document_embeddings = get_embeddings(documents)

        dimension = document_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(document_embeddings)

        metadata = all_data
        faiss.write_index(index, "shl_faiss_index.index")
        with open("shl_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    return index, metadata

# Search function
def search_assessments(query_text, index, metadata, k=10):
    query_vector = get_embeddings([query_text], task_type="retrieval_query")
    distances, indices = index.search(query_vector, k)
    return [metadata[i] for i in indices[0]]


def gemini_llm(Query):
    retriever = index.as_retriever()
    # Set up system prompt
    system_prompt = (
        f"You are an SHL assessment recommendation assistant. Based on the provided job description and context,"
        f" select the most suitable assessments for the role. Prioritize assessments that match test types and skills."
        f" Return the results in a clean table with columns: Assessment Name, Test Types, Duration, Remote Support, Adaptive Support, and URL."
        f" Do not include any extra explanation or intro text. Just the table."
        "\n\nContext:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{input}")]
    )

    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY,convert_system_message_to_human=True), prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Ensure the vector dimensions match the FAISS index

    # Invoke the RAG chain
    answer = rag_chain.invoke(
        {"input": f"Query: {Query}"}
    )

    # Inspect the result structure
    return answer["answer"]

# --- Streamlit UI ---
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🔍 SHL Assessment Recommender (RAG + Gemini + FAISS)")

query = st.text_input("Enter job role or requirement:", placeholder="e.g. Frontend developer with JavaScript")

if query:
    with st.spinner("Searching assessments..."):
        index, metadata = load_index_and_metadata()
        enhanced_query = query_enhancer(query)

        results = search_assessments(enhanced_query, index, metadata, k=10)

        if results:
            st.subheader("📋 Top Matching Assessments")
            table_data = [
                {
                    "Assessment Name": r["name"],
                    "URL": r["url"],
                    "Remote Support": r["remote_support"],
                    "Adaptive/IRT": r.get("adaptive_support", "N/A"),
                    "Test Types": ", ".join(r.get("test_types", [])),
                    "Duration (mins)": r.get("duration", "N/A")
                }
                for r in results
            ]


            st.dataframe(table_data, use_container_width=True)
        else:
            st.warning("No assessments found.")
