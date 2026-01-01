# to run: fastapi dev main.py
from fastapi import FastAPI

from wiki import Wiki
from ai import generate_answer

app = FastAPI(title="AI Knowledge Assistant")

wiki = Wiki()

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Knowledge Assistant!"}

@app.get("/ask")
async def ask_question(query: str):
    wiki.get_wikipedia_content(query)

    if wiki.text:
        wiki.build_faiss_index()
    else:
        return {
            "query": query,
            "answer": "No Wikipedia content found for this query.",
            "sources": []
        }

    results = wiki.search_faiss_index(query)  # retrieval

    if not results:
        return {
            "query": query,
            "answer": "No relevant knowledge found to answer the query.",
            "sources": []
        }

    context = " ".join([r["text"] for r in results])
    answer = await generate_answer(query, context)
    
    seen = set()
    sources = []
    for r in results:
        title = r["metadata"]["title"]
        url = r["metadata"]["url"]
        if title not in seen:
            sources.append({"title": title, "url": url})
            seen.add(title)

    return {
        "query": query, 
        "answer": answer,
        "sources": sources
        }