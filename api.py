from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from fastapi.middleware.cors import CORSMiddleware

# Add this after `app = FastAPI()`

# --- Load ENV ---
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# --- FastAPI Setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or use ["http://localhost:3000"] to be strict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Input Schema ---
class QueryRequest(BaseModel):
    query: str

# --- Embedding ---
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

# --- Enhance Query ---
def query_enhancer(query):
    prompt = f"""
    You are a query enhancer for SHL assessments.
    Given a vague or general input like: "{query}", convert it into a concise and structured job description.
    Format: [Job Title] [Duration in mins] [Comma-separated key skills]
    Example: Frontend Developer 40 mins js, html, css, react
    ⚠️ Only return a single line like the example. No extra text.
    """
    answer = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return answer.text.strip()

# --- Load Index ---
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

# --- Search Logic ---
def search_assessments(query_text, index, metadata, k=10):
    query_vector = get_embeddings([query_text], task_type="retrieval_query")
    distances, indices = index.search(query_vector, k)
    return [metadata[i] for i in indices[0]]

# --- API Endpoint ---
@app.post("/recommend")
def recommend_assessments(payload: QueryRequest):
    index, metadata = load_index_and_metadata()
    enhanced_query = query_enhancer(payload.query)
    results = search_assessments(enhanced_query, index, metadata, k=10)

    formatted_results = [
        {
            "name": r["name"],
            "url": r["url"],
            "remote_support": r["remote_support"],
            "adaptive_support": r.get("adaptive_support", "N/A"),
            "test_types": r.get("test_types", []),
            "duration": r.get("duration", "N/A")
        }
        for r in results
    ]

    return {"query": enhanced_query, "results": formatted_results}
