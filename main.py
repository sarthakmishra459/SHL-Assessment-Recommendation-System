import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tabulate import tabulate

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

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

# Load or generate index
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
    documents = [d["name"] + " " + " ".join(d.get("test_types", [])) for d in all_data]
    document_embeddings = get_embeddings(documents)

    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(document_embeddings)

    metadata = all_data

    # Save index and metadata
    faiss.write_index(index, "shl_faiss_index.index")
    with open("shl_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def search_assessments(query_text, k=10):
    query_vector = get_embeddings([query_text], task_type="retrieval_query")
    distances, indices = index.search(query_vector, k)
    return [metadata[i] for i in indices[0]]


results = search_assessments("industrial Engineering")

# Prepare rows for tabulate
table_data = [
    [
        r["name"],
        r["url"],
        r["remote_support"],
        r["adaptive_support"],
        ", ".join(r["test_types"])
    ]
    for r in results
]

# Define headers
headers = ["Assessment Name", "URL", "Remote Support", "Adaptive/IRT", "Test Types"]

# Print table
print(tabulate(table_data, headers=headers, tablefmt="grid"))
