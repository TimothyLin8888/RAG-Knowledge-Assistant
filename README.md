# RAG-Knowledge-Assistant
An AI-powered knowledge assistant that answers user questions by retrieving relevant information from Wikipedia and generating context-aware responses using a Retrieval-Augmented Generation (RAG) pipeline.
Built with FastAPI, FAISS, and OpenAI, this project demonstrates modern AI system design, semantic search, and backend API development.

## Features
- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with large language model generation to produce grounded, factual answers.

- **Wikipedia-Powered Knowledge Base**: Dynamically fetches and processes Wikipedia articles relevant to user queries.

- **Semantic Search with Vector Embeddings**: Uses OpenAI embeddings and FAISS vector indexing to retrieve the most relevant content.

- **FastAPI Backend**: Asynchronous REST API for handling queries efficiently.

- **Source Attribution**: Returns Wikipedia sources used to generate each answer.


## How it Works
1. **Query Input**: A user submits a question through the API or web interface.
2. **Document Retrieval**: Relevant Wikipedia articles are fetched and chunked into smaller text segments.
3. **Embedding & Indexing**: Text chunks are converted into vector embeddings and stored in a FAISS index.
4. **Semantic Search**: The user query is embedded and compared against stored vectors to find the most relevant context.
5. **Answer Generation**: The retrieved context is passed to an OpenAI language model to generate a grounded response.

## Technologies
- Python
- FastAPI
- OpenAI API (Embeddings + LLMs)
- FAISS (Vector Database)
- Wikipedia API

## Feature Improvements
- Persistent vector storage (no re-indexing per request)

- Multi-source ingestion (PDFs, research papers, web articles)

- User memory & conversation history

- LangChain-based memory and agent orchestration

- Improved frontend UX

- Authentication & multi-user support
