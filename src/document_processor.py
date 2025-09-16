import pytesseract
from PIL import Image
import fitz  # PyMuPDF for PDF processing
import os
import tempfile
from pathlib import Path
import json
import uuid
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)
embedding_model = TextEmbedding()

# Constants
VECTOR_SIZE = 384  # FastEmbed default

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end == text_length:
            break
            
        start += chunk_size - overlap
        
    return chunks

def process_image(file_path, filename, doc_id, user_id):
    """Process an image file with OCR"""
    try:
        # Add more detailed error handling and debugging
        print(f"Processing image: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
            
        # Try to open with PIL with explicit format detection
        try:
            img = Image.open(file_path)
            # Force load the image to verify it can be processed
            img.load()
            print(f"Image format: {img.format}, Size: {img.size}, Mode: {img.mode}")
        except Exception as img_error:
            # If PIL fails, try converting the image using another method
            print(f"PIL failed to open image: {str(img_error)}")
            raise Exception(f"Failed to open image: {str(img_error)}")
        
        # Extract text using OCR
        text = pytesseract.image_to_string(img)
        
        # Check if OCR extracted any text
        if not text.strip():
            print("Warning: OCR extracted no text from the image")
            text = "[No text could be extracted from this image]"
        
        # Create chunks from the OCR text
        chunks = chunk_text(text)
        
        # Create metadata for each chunk
        processed_chunks = []
        for idx, chunk in enumerate(chunks):
            processed_chunks.append({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_id": idx,
                "chunk_text": chunk
            })
        
        # Upload chunks to vector database
        upload_chunks_to_qdrant(processed_chunks, user_id)
        
        return processed_chunks
    except Exception as e:
        print(f"Detailed error processing image: {type(e).__name__}: {str(e)}")
        raise Exception(f"Error processing image: {str(e)}")


def process_pdf(file_path, filename, doc_id, user_id):
    """Process a PDF file"""
    try:
        # Check if file exists and has content
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        file_size = os.path.getsize(file_path)
        print(f"PDF file size: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError(f"PDF file is empty: {file_path}")
        
        # Try to open the PDF with more detailed error handling
        try:
            print(f"Opening PDF: {file_path}")
            pdf_document = fitz.open(file_path)
            print(f"PDF opened successfully. Pages: {len(pdf_document)}")
        except Exception as pdf_error:
            print(f"Error opening PDF with PyMuPDF: {type(pdf_error).__name__}: {str(pdf_error)}")
            
            # Try an alternative approach - read the file in binary mode first
            with open(file_path, 'rb') as pdf_file:
                pdf_content = pdf_file.read()
                
            if len(pdf_content) == 0:
                raise ValueError(f"PDF file content is empty")
                
            # Create a temporary file with the content
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_pdf.write(pdf_content)
            temp_pdf.close()
            
            try:
                pdf_document = fitz.open(temp_pdf.name)
                print(f"PDF opened successfully via temp file. Pages: {len(pdf_document)}")
            except Exception as temp_pdf_error:
                os.unlink(temp_pdf.name)
                raise Exception(f"Failed to open PDF even with alternative method: {str(temp_pdf_error)}")
        
        # Extract text from each page
        all_text = ""
        for page_num in range(len(pdf_document)):
            try:
                page = pdf_document[page_num]
                page_text = page.get_text()
                all_text += page_text
                print(f"Extracted {len(page_text)} characters from page {page_num+1}")
            except Exception as page_error:
                print(f"Error extracting text from page {page_num+1}: {str(page_error)}")
        
        # Check if we got any text
        if not all_text.strip():
            print("Warning: No text extracted from PDF. It might be scanned or contain only images.")
            # Try to extract text from images in the PDF
            all_text = "[This PDF appears to contain no extractable text. It may be scanned or image-based.]"
        
        # Create chunks from the PDF text
        chunks = chunk_text(all_text)
        print(f"Created {len(chunks)} text chunks from PDF")
        
        # Create metadata for each chunk
        processed_chunks = []
        for idx, chunk in enumerate(chunks):
            processed_chunks.append({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_id": idx,
                "chunk_text": chunk
            })
        
        # Upload chunks to vector database
        upload_chunks_to_qdrant(processed_chunks, user_id)
        
        # Clean up any temporary files
        if 'temp_pdf' in locals() and os.path.exists(temp_pdf.name):
            os.unlink(temp_pdf.name)
            
        return processed_chunks
    except Exception as e:
        print(f"Detailed PDF processing error: {type(e).__name__}: {str(e)}")
        raise Exception(f"Error processing PDF: {str(e)}")


def process_text_file(file_path, filename, doc_id, user_id):
    """Process a text file"""
    try:
        print(f"Processing text file: {filename}")
        
        # Check if file exists and has content
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")
            
        file_size = os.path.getsize(file_path)
        print(f"Text file size: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError(f"Text file is empty: {file_path}")
        
        # Read the text file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text_content = f.read()
            
        print(f"Read {len(text_content)} characters from text file")
        
        # Create chunks from the text
        chunks = chunk_text(text_content)
        print(f"Created {len(chunks)} text chunks")
        
        # Create metadata for each chunk
        processed_chunks = []
        for idx, chunk in enumerate(chunks):
            processed_chunks.append({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_id": idx,
                "chunk_text": chunk
            })
        
        # Upload chunks to vector database
        upload_chunks_to_qdrant(processed_chunks, user_id)
        
        return processed_chunks
    except Exception as e:
        print(f"Error processing text file: {type(e).__name__}: {str(e)}")
        raise Exception(f"Error processing text file: {str(e)}")

def process_document(file_path, filename, doc_id, user_id):
    """Process a document based on its file type"""
    try:
        print(f"Processing document: {filename} at path: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Get file extension
        file_extension = Path(filename).suffix.lower()
        print(f"Detected file extension: {file_extension}")
        
        # Handle text files
        if file_extension in ['.txt', '.md', '.csv', '.json']:
            print(f"Processing as text file")
            return process_text_file(file_path, filename, doc_id, user_id)
        # Handle image files
        elif file_extension in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif']:
            print(f"Processing as image file")
            return process_image(file_path, filename, doc_id, user_id)
        # Handle PDF files
        elif file_extension == '.pdf':
            print(f"Processing as PDF file")
            return process_pdf(file_path, filename, doc_id, user_id)
        # Try to process as text file for unknown extensions
        else:
            print(f"Unrecognized extension: {file_extension}, attempting to process as text file")
            try:
                return process_text_file(file_path, filename, doc_id, user_id)
            except Exception as text_error:
                print(f"Failed to process as text file: {str(text_error)}")
                
                # Try as image
                try:
                    print("Attempting to process as image instead")
                    return process_image(file_path, filename, doc_id, user_id)
                except Exception as img_error:
                    print(f"Failed to process as image: {str(img_error)}")
                    raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        print(f"Error in process_document: {type(e).__name__}: {str(e)}")
        raise Exception(f"Error processing document: {str(e)}")



def create_user_collection(user_id):
    """Create a collection for a specific user if it doesn't exist"""
    collection_name = f"user_{user_id}_docs"
    
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
    
    return collection_name

def get_qdrant_client():
    """Get a Qdrant client instance"""
    try:
        # Use the global client that was initialized at the top of the file
        global client
        if client is None:
            client = QdrantClient(host="localhost", port=6333)
        return client
    except Exception as e:
        print(f"Error getting Qdrant client: {type(e).__name__}: {str(e)}")
        return None

def upload_chunks_to_qdrant(chunks, user_id):
    """Upload chunks to the user's Qdrant collection"""
    # Get or create the user's collection
    collection_name = f"user_{user_id}_docs"
    
    try:
        print(f"Uploading {len(chunks)} chunks to Qdrant for user {user_id}")
        print(f"Collection name: {collection_name}")
        
        # Debug Qdrant connection
        try:
            print("Testing Qdrant connection...")
            collections = client.get_collections().collections
            print(f"Available collections: {[c.name for c in collections]}")
        except Exception as conn_error:
            print(f"ERROR: Could not connect to Qdrant: {type(conn_error).__name__}: {str(conn_error)}")
            raise
        
        # Create collection if it doesn't exist
        try:
            print(f"Creating collection {collection_name} if it doesn't exist")
            create_user_collection(user_id)
            print(f"Collection {collection_name} is ready")
        except Exception as coll_error:
            print(f"ERROR: Could not create collection: {type(coll_error).__name__}: {str(coll_error)}")
            raise
        
        # Initialize embedding model - use the global one that was already initialized
        global embedding_model
        try:
            if embedding_model is None:
                print("Initializing embedding model...")
                embedding_model = TextEmbedding()
            print("Embedding model is ready")
        except Exception as emb_error:
            print(f"ERROR: Could not initialize embedding model: {type(emb_error).__name__}: {str(emb_error)}")
            raise
        
        # Prepare points for upload
        points = []
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                print(f"Generating embedding for chunk {i+1}/{len(chunks)}")
                chunk_text = chunk["chunk_text"]
                if not chunk_text or len(chunk_text.strip()) == 0:
                    print(f"Warning: Empty text in chunk {i+1}, skipping")
                    continue
                
                print(f"Chunk {i+1} text sample: {chunk_text[:50]}...")
                # Convert generator to list to avoid 'generator' has no len() error
                embeddings = list(embedding_model.embed([chunk_text]))
                
                if embeddings is None or len(embeddings) == 0 or embeddings[0] is None:
                    print(f"Warning: Failed to generate embedding for chunk {i+1}, skipping")
                    continue
                
                # Create point with an integer ID (Qdrant requires either UUID or integer)
                # Using a simple integer is more reliable than UUID strings
                point = PointStruct(
                    id=i,  # Use the loop counter as a simple integer ID
                    vector=embeddings[0].tolist(),
                    payload={
                        "doc_id": chunk["doc_id"],
                        "filename": chunk["filename"],
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk_text,
                        "original_id": f"{chunk['doc_id']}_{chunk['chunk_id']}"  # Store original ID in payload for reference
                    }
                )
                points.append(point)
                print(f"Created point for chunk {i+1} with ID {point.id}")
            except Exception as chunk_error:
                print(f"Error processing chunk {i+1}: {type(chunk_error).__name__}: {str(chunk_error)}")
                # Continue with other chunks instead of failing completely
                continue
        
        if not points:
            print("Warning: No valid points to upload to Qdrant")
            return 0
            
        print(f"Uploading {len(points)} points to Qdrant collection {collection_name}")
        
        # Upload points to Qdrant
        try:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            print("Successfully uploaded points to Qdrant")
            print(f"Upload complete! {len(points)} chunks added to collection.")
            return len(points)
        except Exception as upsert_error:
            print(f"ERROR: Failed to upsert points to Qdrant: {type(upsert_error).__name__}: {str(upsert_error)}")
            raise
    except Exception as e:
        print(f"Error in upload_chunks_to_qdrant: {type(e).__name__}: {str(e)}")
        raise Exception(f"Failed to upload chunks to vector database: {str(e)}")
