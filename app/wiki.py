import wikipedia
from langchain_openai import OpenAIEmbeddings
import numpy as np
import faiss

class Wiki:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.index = None
        
        self.text = [] # summaries
        self.metadata = [] # title, url


    def get_wikipedia_content(self, query: str, max_results: int =10) -> list[dict]:
        """Fetches a short summary from Wikipedia for the query.
        
        query: The question that the user asks to be found.
        """
        try:
            search_results = wikipedia.search(query, results=max_results)
            seen_titles = set()
            self.texts.clear()
            self.metadatas.clear()

            for title in search_results:
                # Deduplicate titles
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                try:
                    page = wikipedia.page(title)
                    content = page.content  # full page content
                    url = page.url

                    # Chunk content into 300-word segments
                    words = content.split()
                    chunk_size = 300
                    for i in range(0, len(words), chunk_size):
                        chunk_text = " ".join(words[i:i+chunk_size])
                        self.texts.append(chunk_text)
                        self.metadatas.append({"title": page.title, "url": url})

                except wikipedia.DisambiguationError as e:
                    # Skip disambiguation pages
                    continue
                except wikipedia.PageError as e:
                    continue

            return [{"title": m["title"], "url": m["url"]} for m in self.metadatas]

        except Exception as e:
            return [{"error": str(e)}]
        
    def build_faiss_index(self):
        vectors = self.embeddings.embed_documents(self.text)
        vectors = np.array(vectors).astype("float32")

        print(vectors.shape)
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)

    def search_faiss_index(self, query: str, top_k: int = 5):
        query_vector = self.embeddings.embed_query(query)
        query_vector = np.array([query_vector]).astype("float32")

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                "text": self.text[idx],
                "metadata": self.metadata[idx],
                "distance": float(dist)
            })

        return results
    
    # def generate_embeddings(self, articles):
    #     chunks = []
    #     for article in articles:
    #         content = article["content"]
    #         embeddings = self.embeddings.embed_text(content)
    #         chunks.append({"text": content, "embedding": embeddings})
    #     return chunks

if __name__ == "__main__":
    wiki = Wiki()

    # Fetch Wikipedia content for a given query
    query = "Who is the current president of the United States?"
    articles = wiki.get_wikipedia_content(query)
    print("Fetched Articles:")
    for article in articles:
        print(f"Title: {article['title']}, URL: {article['url']}")

    # Build FAISS index using the fetched articles
    wiki.build_faiss_index()

    # Use the FAISS index to search for relevant articles
    search_query = "current president"
    search_results = wiki.search_faiss_index(search_query)
    
    print("\nSearch Results:")
    for result in search_results:
        print(f"Text: {result['text']}, Metadata: {result['metadata']}, Distance: {result['distance']}")