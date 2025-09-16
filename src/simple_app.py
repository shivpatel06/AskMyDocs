import gradio as gr
import os
import tempfile
from pathlib import Path
import uuid
from datetime import datetime

# Import LlamaIndex components
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# Import our custom modules
from document_processor import process_document, upload_chunks_to_qdrant
from chat_history import save_chat_message, get_chat_history, save_document_record, get_user_documents
from aws_cognito import sign_up, sign_in, confirm_sign_up, local_auth_enabled

# Set up LlamaIndex to use FastEmbed for embeddings and Llama 3 via Ollama for LLM
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="llama3", request_timeout=120.0)

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Global variables
UPLOAD_FOLDER = Path("./uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
current_user_id = None  # Will be set after login
user_sessions = {}  # Store active user sessions

def get_user_collection(user_id):
    """Get or create a collection for a specific user"""
    collection_name = f"user_{user_id}_docs"
    
    if not client.collection_exists(collection_name=collection_name):
        # If no documents have been uploaded yet, use the default collection
        if client.collection_exists(collection_name="ocr_chunks"):
            return "ocr_chunks"
        else:
            return None
    
    return collection_name

def get_query_engine(user_id):
    """Get a query engine for a specific user"""
    try:
        print(f"Getting query engine for user {user_id}")
        
        # Get collection name for this user
        collection_name = f"user_{user_id}_docs"
        print(f"Looking for collection: {collection_name}")
        
        # Check if collection exists
        try:
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name not in collection_names:
                print(f"Collection {collection_name} does not exist")
                return None
                
            # Check if collection has points
            collection_info = client.get_collection(collection_name)
            if collection_info.points_count == 0:
                print(f"Collection {collection_name} exists but is empty")
                return None
                
            print(f"Found collection {collection_name} with {collection_info.points_count} points")
        except Exception as coll_error:
            print(f"Error checking collection: {type(coll_error).__name__}: {str(coll_error)}")
            return None
        
        # Create vector store
        try:
            print("Creating vector store...")
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name
            )
            print("Vector store created successfully")
        except Exception as vs_error:
            print(f"Error creating vector store: {type(vs_error).__name__}: {str(vs_error)}")
            return None
        
        # Create index
        try:
            print("Creating vector store index...")
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            print("Vector store index created successfully")
        except Exception as idx_error:
            print(f"Error creating index: {type(idx_error).__name__}: {str(idx_error)}")
            return None
        
        # Create query engine
        try:
            print("Creating query engine...")
            query_engine = index.as_query_engine(
                llm=Settings.llm,
                similarity_top_k=3
            )
            print("Query engine created successfully")
            return query_engine
        except Exception as qe_error:
            print(f"Error creating query engine: {type(qe_error).__name__}: {str(qe_error)}")
            return None
    except Exception as e:
        print(f"Unexpected error in get_query_engine: {type(e).__name__}: {str(e)}")
        return None

def process_uploaded_file(files):
    """Process and upload one or more documents"""
    global current_user_id
    
    if files is None or len(files) == 0:
        return "No files uploaded"
    
    if not current_user_id:
        return "You must be logged in to upload documents"
    
    results = []
    total_chunks = 0
    
    for file in files:
        try:
            # Gradio provides the file path directly - no need to create temp file
            file_path = file.name
            file_name = os.path.basename(file_path)
            
            # Process the document directly from the Gradio temp file
            doc_id = str(uuid.uuid4())
            chunks = process_document(file_path, file_name, doc_id, current_user_id)
            
            if not chunks or len(chunks) == 0:
                results.append(f"⚠️ {file_name}: No text could be extracted")
                continue
            
            # Upload chunks to Qdrant
            upload_chunks_to_qdrant(chunks, current_user_id)
            
            # Save document record
            save_document_record(current_user_id, doc_id, file_name)
            
            total_chunks += len(chunks)
            results.append(f"✅ {file_name}: {len(chunks)} chunks extracted")
            
        except Exception as e:
            results.append(f"❌ {file.name if hasattr(file, 'name') else 'Unknown file'}: Error - {str(e)}")
    
    summary = f"Processed {len(files)} document(s) with {total_chunks} total chunks extracted."
    return summary + "\n\n" + "\n".join(results)

def display_user_message(message, history):
    """Display user message immediately in chat"""
    if not message.strip():
        return history, ""
    
    # Add user message to history with empty response
    updated_history = history + [(message, "")]
    return updated_history, ""

def get_ai_response(history):
    """Generate AI response for the last user message"""
    if not history:
        return history
    
    # Get the last message (should be user message with empty response)
    user_message, current_response = history[-1]
    
    # If there's already a response, don't regenerate
    if current_response and current_response.strip():
        return history
    
    try:
        print(f"Processing chat message: {user_message}")
        
        # Get query engine
        query_engine = get_query_engine(current_user_id)
        if not query_engine:
            print("No query engine available - user needs to upload documents first")
            history[-1] = (user_message, "Please upload a document first. Go to the 'Upload Documents' tab to add a document.")
            return history
        
        # Query the documents
        try:
            print("Querying documents...")
            response = query_engine.query(user_message)
            response_text = str(response)
            print(f"Got response: {response_text[:100]}...")
        except Exception as query_error:
            print(f"Error querying documents: {type(query_error).__name__}: {str(query_error)}")
            history[-1] = (user_message, f"Error processing your query: {str(query_error)}")
            return history
        
        # Save chat history
        try:
            save_chat_message(current_user_id, "user", user_message, datetime.now().isoformat())
            save_chat_message(current_user_id, "assistant", response_text, datetime.now().isoformat())
            print("Chat history saved successfully")
        except Exception as history_error:
            print(f"Error saving chat history: {str(history_error)}")
        
        # Update the response
        history[-1] = (user_message, response_text)
        return history
        
    except Exception as e:
        print(f"Unexpected error in get_ai_response: {type(e).__name__}: {str(e)}")
        history[-1] = (user_message, f"An error occurred while processing your message: {str(e)}")
        return history

def chat_with_documents(message, history):
    """Chat with documents using RAG"""
    try:
        new_history, _ = display_user_message(message, history)
        return get_ai_response(new_history)
    except Exception as e:
        print(f"Unexpected error in chat_with_documents: {type(e).__name__}: {str(e)}")
        return [(message, f"An error occurred while processing your message: {str(e)}")]

def list_documents():
    """List the user's documents as markdown"""
    documents = get_user_documents(current_user_id)
    if not documents:
        return "No documents found"
    
    formatted = []
    for doc in documents:
        time = datetime.fromisoformat(doc["upload_timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        formatted.append(f"**{doc['filename']}** (ID: {doc['doc_id']}, Uploaded: {time})")
    
    return "\n".join(formatted)

def list_documents_as_dataframe():
    """List the user's documents as a dataframe"""
    documents = get_user_documents(current_user_id)
    if not documents:
        return [["No documents found", "", ""]]
    
    rows = []
    for doc in documents:
        time = datetime.fromisoformat(doc["upload_timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        rows.append([doc["doc_id"], doc["filename"], time])
    
    return rows

def get_document_history_as_dataframe():
    """Get document history as a dataframe"""
    documents = get_user_documents(current_user_id)
    if not documents:
        return [["No documents found", "", ""]]
    
    rows = []
    for doc in documents:
        time = datetime.fromisoformat(doc["upload_timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        rows.append([doc["doc_id"], doc["filename"], time])
    
    return rows

def get_chat_history_as_dataframe():
    """Get chat history as a dataframe"""
    history = get_chat_history(current_user_id)
    if not history:
        return [["No chat history", "", ""]]
    
    rows = []
    for msg in history:
        time = datetime.fromisoformat(msg["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        role = "You" if msg.get("role", msg.get("message_type")) == "user" else "Assistant"
        rows.append([time, role, msg["content"]])
    
    return rows

def show_chat_history():
    """Format chat history for display"""
    history = get_chat_history(current_user_id)
    formatted = []
    
    for msg in history:
        role = "You" if msg.get("role", msg.get("message_type")) == "user" else "Assistant"
        time = datetime.fromisoformat(msg["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        formatted.append(f"**{role}** ({time}):\n{msg['content']}\n")
    
    return "\n".join(formatted)

# Authentication functions
def login(username, password):
    """Log in a user with AWS Cognito or local authentication"""
    global current_user_id
    
    if not username or not password:
        return "Please enter both username and password", None, gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
    
    result = sign_in(username, password)
    
    if result["success"]:
        current_user_id = result["user_id"]
        user_sessions[current_user_id] = {
            "username": username,
            "logged_in": True,
            "tokens": result.get("tokens", {})
        }
        return f"Welcome, {username}!", None, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
    else:
        return f"Login failed: {result['message']}", None, gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)

def register(username, password, email):
    """Register a new user with AWS Cognito or local authentication"""
    if not username or not password or not email:
        return "Please fill in all fields", None, gr.update(visible=True)
    
    result = sign_up(username, password, email)
    
    if result["success"]:
        return f"Registration successful! {result['message']}", None, gr.update(visible=True)
    else:
        return f"Registration failed: {result['message']}", None, gr.update(visible=True)

def confirm_registration(username, confirmation_code):
    """Confirm user registration with verification code"""
    if not username or not confirmation_code:
        return "Please enter both username and confirmation code", gr.update(visible=True)
    
    result = confirm_sign_up(username, confirmation_code)
    
    if result["success"]:
        return f"Confirmation successful! You can now log in.", gr.update(visible=True)
    else:
        return f"Confirmation failed: {result['message']}", gr.update(visible=True)

def logout():
    """Log out the current user"""
    global current_user_id
    
    if current_user_id and current_user_id in user_sessions:
        del user_sessions[current_user_id]
    
    current_user_id = None
    return "You have been logged out.", None, gr.update(visible=True), gr.update(visible=False)

def check_login():
    """Check if user is logged in and return appropriate visibility flags"""
    if current_user_id and current_user_id in user_sessions and user_sessions[current_user_id]["logged_in"]:
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False)

# Create the Gradio interface with authentication
with gr.Blocks(title="RAG Chatbot", theme=gr.themes.Soft(primary_hue="blue")) as demo:
    with gr.Column(elem_id="container"):
        gr.Markdown("""
        # 📚 AskMyDocs
        ### Chat with your documents using Retrieval-Augmented Generation
        """)
        
        # Authentication UI
        with gr.Group(visible=True, elem_id="auth-container") as login_group:
            with gr.Tabs() as auth_tabs:
                with gr.TabItem("Login", elem_id="login-tab"):
                    with gr.Column():
                        username_login = gr.Textbox(label="Username", placeholder="Enter your username", elem_id="login-username")
                        password_login = gr.Textbox(label="Password", type="password", placeholder="Enter your password", elem_id="login-password")
                        login_button = gr.Button("Login", variant="primary", elem_id="login-button")
                        login_message = gr.Textbox(label="Status", interactive=False, visible=False, elem_id="login-message")
                
                with gr.TabItem("Register", elem_id="register-tab"):
                    with gr.Column():
                        username_register = gr.Textbox(label="Username", placeholder="Choose a username", elem_id="register-username")
                        password_register = gr.Textbox(label="Password", type="password", placeholder="Choose a secure password", elem_id="register-password")
                        email_register = gr.Textbox(label="Email", placeholder="Enter your email address", elem_id="register-email")
                        register_button = gr.Button("Register", variant="primary", elem_id="register-button")
                        register_message = gr.Textbox(label="Status", interactive=False, visible=False, elem_id="register-message")
                
                with gr.TabItem("Confirm Registration", elem_id="confirm-tab"):
                    with gr.Column():
                        username_confirm = gr.Textbox(label="Username", placeholder="Enter your username", elem_id="confirm-username")
                        confirmation_code = gr.Textbox(label="Confirmation Code", placeholder="Enter the code from your email", elem_id="confirm-code")
                        confirm_button = gr.Button("Confirm Registration", variant="primary", elem_id="confirm-button")
                        confirm_message = gr.Textbox(label="Status", interactive=False, visible=False, elem_id="confirm-message")
    
    # Main Application UI (hidden until login)
    with gr.Group(visible=False, elem_id="app-container") as app_group:
        with gr.Row(elem_id="header-row"):
            with gr.Column(scale=5):
                gr.Markdown("### 👋 Welcome to your RAG Chatbot Dashboard")
            with gr.Column(scale=1, min_width=120):
                logout_button = gr.Button("Logout", variant="secondary", elem_id="logout-button")
        
        with gr.Tabs(elem_id="main-tabs") as tabs:
            with gr.TabItem("📄 Upload Documents", elem_id="upload-tab"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        gr.Markdown("### Add New Documents")
                        file_upload = gr.File(
                            label="Select Document", 
                            file_types=["pdf", "png", "jpg", "jpeg", "tiff"], 
                            file_count="multiple",
                            elem_id="file-upload"
                        )
                        upload_button = gr.Button("Process Document", variant="primary", elem_id="upload-button")
                        upload_status = gr.Textbox(label="Status", interactive=False, elem_id="upload-status")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Your Document Library")
                        documents_list = gr.Dataframe(
                            headers=["Document ID", "Filename", "Upload Time"],
                            label="Your Documents",
                            interactive=False,
                            elem_id="documents-list",
                            wrap=True
                        )
                        refresh_docs_button = gr.Button("Refresh List", elem_id="refresh-docs-button")
            
            with gr.TabItem("💬 Chat with Documents", elem_id="chat-tab"):
                with gr.Column():
                    gr.Markdown("### Ask questions about your uploaded documents")
                    chatbot = gr.Chatbot(height=500, elem_id="chatbot")
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your Question", 
                            placeholder="Ask something about your documents...",
                            scale=9,
                            elem_id="msg-input",
                            container=False
                        )
                        clear = gr.Button("Clear", scale=1, elem_id="clear-button")
            
            with gr.TabItem("📊 Chat History", elem_id="history-tab"):
                with gr.Column():
                    gr.Markdown("### Your Conversation History")
                    chat_history_list = gr.Dataframe(
                        headers=["Time", "Role", "Message"],
                        label="Chat History",
                        interactive=False,
                        elem_id="chat-history-list",
                        wrap=True,
                        height=400
                    )
                    refresh_history_button = gr.Button("Refresh History", elem_id="refresh-history-button")
            
            with gr.TabItem("📚 Document History", elem_id="doc-history-tab"):
                with gr.Column():
                    gr.Markdown("### Your Document Upload History")
                    doc_history_list = gr.Dataframe(
                        headers=["Document ID", "Filename", "Upload Time"],
                        label="Document History",
                        interactive=False,
                        elem_id="doc-history-list",
                        wrap=True,
                        height=400
                    )
                    refresh_doc_history_button = gr.Button("Refresh History", elem_id="refresh-doc-history-button")
    
    # Set up event handlers for authentication
    login_button.click(
        login,
        inputs=[username_login, password_login],
        outputs=[login_message, password_login, login_group, app_group, login_message]
    )
    
    register_button.click(
        register,
        inputs=[username_register, password_register, email_register],
        outputs=[register_message, password_register, register_message]
    )
    
    confirm_button.click(
        confirm_registration,
        inputs=[username_confirm, confirmation_code],
        outputs=[confirm_message, confirm_message]
    )
    
    logout_button.click(
        logout,
        inputs=[],
        outputs=[login_message, password_login, login_group, app_group]
    )
    
    # Set up event handlers for the main application
    upload_button.click(
        process_uploaded_file,
        inputs=[file_upload],
        outputs=[upload_status]
    ).then(
        list_documents_as_dataframe,
        inputs=[],
        outputs=[documents_list]
    )
    
    refresh_docs_button.click(
        list_documents_as_dataframe,
        inputs=[],
        outputs=[documents_list]
    )
    
    msg.submit(
        display_user_message,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    ).then(
        get_ai_response,
        inputs=[chatbot],
        outputs=[chatbot]
    )
    
    clear.click(
        lambda: [],
        None,
        [chatbot]
    )
    
    refresh_history_button.click(
        get_chat_history_as_dataframe,
        inputs=[],
        outputs=[chat_history_list]
    )
    
    refresh_doc_history_button.click(
        get_document_history_as_dataframe,
        inputs=[],
        outputs=[doc_history_list]
    )
    
    # Auto-refresh document and chat history when app loads
    demo.load(
        list_documents_as_dataframe,
        inputs=None,
        outputs=[documents_list]
    )

if __name__ == "__main__":
    # If using local auth, set a default user for testing
    if local_auth_enabled():
        print("Using local authentication with default user")
        current_user_id = "default_user"
        user_sessions[current_user_id] = {
            "username": "default_user",
            "logged_in": True,
            "tokens": {}
        }
    
    demo.launch(share=True)
