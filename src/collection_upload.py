import json
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastembed import TextEmbedding

# ---- 1. Connect to Qdrant ----
client = QdrantClient("localhost", port=6333)

COLLECTION_NAME = "ocr_chunks"
VECTOR_SIZE = 384  # FastEmbed default

# ---- 2. Delete collection if it exists ----
if client.collection_exists(collection_name=COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    client.delete_collection(collection_name=COLLECTION_NAME)

# ---- 3. Recreate the collection ----
print(f"Creating collection: {COLLECTION_NAME}")
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

# ---- 4. Load your OCR chunks ----
with open("src/ocr_chunks.json") as f:
    chunks = json.load(f)

texts = [c["chunk_text"] for c in chunks]
metadatas = [{"filename": c["filename"], "chunk_id": c["chunk_id"]} for c in chunks]

# ---- 5. Generate embeddings ----
print("Generating embeddings...")
embedding_model = TextEmbedding()
embeddings = list(embedding_model.embed(texts))

# ---- 6. Prepare points for upload ----
print("Preparing points for upload...")
points = [
    PointStruct(
        id=i,
        vector=embeddings[i],
        payload={**metadatas[i], "text": texts[i]}
    )
    for i in range(len(texts))
]

# ---- 7. Upload points to Qdrant ----
print(f"Uploading {len(points)} points to Qdrant...")
client.upsert(
    collection_name=COLLECTION_NAME,
    wait=True,
    points=points
)

print("Upload complete! Your collection is reset and repopulated.")

