# 🔍 SHL Assessment Recommendation System

This project is an **intelligent assessment recommendation system** powered by Google Gemini Embeddings and FAISS. It suggests the most relevant SHL assessments (individual or pre-packaged) based on natural language job descriptions or queries.

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Embedding Model**: `text-embedding-004` (Gemini)
- **Vector Store**: FAISS
- **Web Scraping**: BeautifulSoup, Requests
- **Others**: NumPy, dotenv

---

## 📁 Project Structure

| File/Folder                  | Description |
|-----------------------------|-------------|
| `api.py`                    | FastAPI backend logic to handle `/recommend` route |
| `app.py`                    | Streamlit frontend interface |
| `scrape_shl_catalog.py`     | Scraper to extract SHL assessments from their catalog |
| `update_shl_data.py`        | Extracts duration and additional info from scraped URLs |
| `shl_assessments_individual.json` | Individual test solutions from SHL |
| `shl_assessments_pre.json`  | Pre-packaged job solutions from SHL |
| `shl_faiss_index.index`     | Saved FAISS vector index of all assessments |
| `shl_metadata.json`         | Metadata mapped to each vector (name, URL, duration, etc.) |
| `sol.ipynb`                 | Jupyter notebook for experimentation and development |
| `test.http`                 | Example REST Client request for testing API locally |
| `requirements.txt`          | Python package dependencies |
| `.gitignore`                | Files to be ignored in version control |

---

## 🚀 How It Works

1. **Web Scraping**  
   SHL assessments are scraped using `scrape_shl_catalog.py`. Two categories are scraped:
   - Individual Test Solutions
   - Pre-packaged Job Solutions

2. **Metadata Enrichment**  
   `update_shl_data.py` is used to extract the duration and other relevant details from the assessment URLs.

3. **Vector Embedding + Indexing**  
   All assessments are embedded using **Google Gemini `text-embedding-004`** and stored in a **FAISS vector index** along with corresponding metadata.

4. **Query Handling**  
   A natural language query is enhanced to extract structured job info using Gemini, then embedded and compared with stored vectors to return the most relevant assessments.

5. **API (FastAPI)**  
   The `/recommend` endpoint accepts a POST or GET request with a query and returns a list of recommended assessments with details like duration, type, and support features.

6. **Frontend (Streamlit)**  
   A simple interface (`app.py`) allows users to input job roles and view recommended assessments visually.

---

## 📬 API Usage

### SHL Assessment Recommendation API
```
GET https://127.0.0.1:8000/recommend?query=Enter_Your_Query_Here
Content-Type: application/json
```
OR
```
POST https://127.0.0.1:8000/recommend
Content-Type: application/json
{
  "query": "Looking for a backend developer with OAuth experience"
}
```
## ✅ Features

🔎 Query enhancer using Gemini to turn vague inputs into structured job descriptions.

📊 Hybrid support for individual and pre-packaged assessments.

⚡ Fast nearest-neighbor search with FAISS.

📂 Fully persistent embeddings and metadata.

🌐 Easy-to-use REST API and frontend.

## 🙌 Author

**Sarthak Mishra**  
GitHub: [@sarthakmishra459](https://github.com/sarthakmishra459)
