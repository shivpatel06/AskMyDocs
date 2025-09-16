import json
from fastembed import TextEmbedding

with open("ocr_chunks.json") as f:
    chunks = json.load(f)

texts = [c["chunk_text"] for c in chunks]
metadatas = [{"filename": c["filename"], "chunk_id": c["chunk_id"]} for c in chunks]

embedding_model = TextEmbedding()
embeddings = list(embedding_model.embed(texts))  # Each is a 384-dim vector