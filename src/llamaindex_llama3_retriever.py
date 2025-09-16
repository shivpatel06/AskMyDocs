from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# 1. Set embedding and LLM to match your pipeline (FastEmbed + Llama 3 via Ollama)
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="llama3", request_timeout=120.0)

# 2. Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(client=client, collection_name="ocr_chunks")

# 3. Build the LlamaIndex index from the vector store
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 4. Create a query engine (retriever + LLM)
query_engine = index.as_query_engine()

# 5. Run a query and print the answer
query = "What does the document say about cigarette warnings?"
response = query_engine.query(query)
print("\nLLM-powered answer:")
print(response)

# 6. (Optional) Print the top retrieved chunks for transparency
print("\nTop retrieved chunks:")
retriever = index.as_retriever()
results = retriever.retrieve(query)
for i, node in enumerate(results, 1):
    print(f"\nResult {i}:")
    print("Score:", node.score)
    print("Text:", node.get_content())
    print("-" * 40)
