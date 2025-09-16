from datetime import datetime
from database import (
    save_chat_message as db_save_chat_message,
    get_chat_history as db_get_chat_history,
    save_document_record as db_save_document_record,
    get_user_documents as db_get_user_documents,
    db_manager
)

def save_chat_message(user_id, message_type, content, timestamp=None):
    """Save a chat message to the database
    
    Args:
        user_id (str): The user ID
        message_type (str): Either 'user' or 'assistant'
        content (str): The message content
        timestamp (str, optional): ISO format timestamp. If None, current time is used.
    """
    # Convert string timestamp to datetime if provided
    if timestamp and isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            # If parsing fails, use current time
            timestamp = None
    
    return db_save_chat_message(user_id, message_type, content, timestamp)

def get_chat_history(user_id, limit=50):
    """Get chat history for a specific user
    
    Args:
        user_id (str): The user ID
        limit (int, optional): Maximum number of messages to retrieve
        
    Returns:
        list: List of chat messages
    """
    return db_get_chat_history(user_id, limit)

def save_document_record(user_id, doc_id, filename):
    """Save a record of an uploaded document
    
    Args:
        user_id (str): The user ID
        doc_id (str): The document ID
        filename (str): The original filename
    """
    return db_save_document_record(user_id, doc_id, filename)

def get_user_documents(user_id):
    """Get all documents uploaded by a specific user
    
    Args:
        user_id (str): The user ID
        
    Returns:
        list: List of document records
    """
    return db_get_user_documents(user_id)

def test_database_connection():
    """Test the database connection"""
    return db_manager.test_connection()
