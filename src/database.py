import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Index, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PostgreSQL configuration
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'ragchatbot')
DB_USER = os.environ.get('DB_USER', '')
DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
DB_SSL_MODE = os.environ.get('DB_SSL_MODE', 'require')

# SQLAlchemy setup
Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    message_type = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Add index for efficient queries
    __table_args__ = (
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
    )

class UserDocument(Base):
    __tablename__ = 'user_documents'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    doc_id = Column(String(255), nullable=False)
    filename = Column(String(500), nullable=False)
    upload_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Add index for efficient queries
    __table_args__ = (
        Index('idx_user_upload', 'user_id', 'upload_timestamp'),
        Index('idx_doc_id', 'doc_id'),
    )

class DatabaseManager:
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.db_type = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection with PostgreSQL fallback to SQLite"""
        # Check if DB_TYPE is explicitly set to sqlite
        db_type = os.getenv('DB_TYPE', '').lower()
        
        if db_type == 'sqlite':
            logger.info("DB_TYPE set to sqlite, using SQLite database")
            self._setup_sqlite()
            return
            
        # Try PostgreSQL first
        try:
            self._setup_postgresql()
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("PostgreSQL database initialized successfully")
            self.db_type = 'postgresql'
            
        except Exception as e:
            logger.warning(f"Failed to initialize PostgreSQL database: {str(e)}")
            logger.info("Falling back to SQLite database")
            try:
                self._setup_sqlite()
                self.db_type = 'sqlite'
            except Exception as sqlite_error:
                logger.error(f"Failed to initialize SQLite fallback: {str(sqlite_error)}")
                raise Exception(f"Both PostgreSQL and SQLite initialization failed. PostgreSQL: {str(e)}, SQLite: {str(sqlite_error)}")
    
    def _setup_postgresql(self):
        """Setup PostgreSQL connection"""
        if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
            raise ValueError("PostgreSQL configuration incomplete. Missing required environment variables: DB_HOST, DB_NAME, DB_USER, DB_PASSWORD")
        
        # Construct PostgreSQL connection string with compatibility settings
        connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        
        # Add SSL mode and compatibility options
        params = []
        if DB_SSL_MODE:
            params.append(f"sslmode={DB_SSL_MODE}")
        
        # Add connection options for older PostgreSQL versions
        params.extend([
            "application_name=ragchatbot",
            "connect_timeout=30",
        ])
        
        if params:
            connection_string += "?" + "&".join(params)
        
        try:
            self.engine = create_engine(
                connection_string,
                pool_size=5,  # Reduced pool size for compatibility
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,  # Recycle connections every hour
                echo=False,
                connect_args={
                    "connect_timeout": 30,
                    "application_name": "ragchatbot"
                }
            )
            
            # Test the connection with a simple query
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version(), current_database()"))
                version, db_name = result.fetchone()
                logger.info(f"PostgreSQL connection established to {DB_HOST}:{DB_PORT}/{db_name}")
                logger.info(f"PostgreSQL version: {version[:50]}...")
            
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
        except Exception as db_error:
            logger.error(f"PostgreSQL connection failed: {type(db_error).__name__}: {str(db_error)}")
            raise db_error
    
    def _setup_sqlite(self):
        """Setup SQLite connection as fallback"""
        sqlite_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chat_history.db')
        connection_string = f"sqlite:///{sqlite_path}"
        
        self.engine = create_engine(
            connection_string,
            pool_pre_ping=True,
            echo=False
        )
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        logger.info(f"SQLite database initialized at {sqlite_path}")
    
    def get_session(self) -> Session:
        """Get a database session"""
        return self.SessionLocal()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False

# Global database manager instance
db_manager = DatabaseManager()

def get_db_session():
    """Get a database session - use this function throughout the application"""
    return db_manager.get_session()

def save_chat_message(user_id: str, message_type: str, content: str, timestamp: datetime = None):
    """Save a chat message to the database
    
    Args:
        user_id (str): The user ID
        message_type (str): Either 'user' or 'assistant'
        content (str): The message content
        timestamp (datetime, optional): Message timestamp. If None, current time is used.
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    try:
        with get_db_session() as session:
            message = ChatMessage(
                user_id=user_id,
                message_type=message_type,
                content=content,
                timestamp=timestamp
            )
            session.add(message)
            session.commit()
            logger.debug(f"Saved chat message for user {user_id}")
    except SQLAlchemyError as e:
        logger.error(f"Error saving chat message: {str(e)}")
        raise Exception(f"Failed to save chat message: {str(e)}")

def get_chat_history(user_id: str, limit: int = 50):
    """Get chat history for a specific user
    
    Args:
        user_id (str): The user ID
        limit (int, optional): Maximum number of messages to retrieve
        
    Returns:
        list: List of chat messages as dictionaries
    """
    try:
        with get_db_session() as session:
            messages = session.query(ChatMessage)\
                .filter(ChatMessage.user_id == user_id)\
                .order_by(ChatMessage.timestamp.asc())\
                .limit(limit)\
                .all()
            
            # Convert to list of dictionaries for compatibility
            result = []
            for msg in messages:
                result.append({
                    'id': msg.id,
                    'user_id': msg.user_id,
                    'role': msg.message_type,  # Keep 'role' for compatibility
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat()
                })
            
            return result
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        return []

def save_document_record(user_id: str, doc_id: str, filename: str):
    """Save a record of an uploaded document
    
    Args:
        user_id (str): The user ID
        doc_id (str): The document ID
        filename (str): The original filename
    """
    try:
        with get_db_session() as session:
            document = UserDocument(
                user_id=user_id,
                doc_id=doc_id,
                filename=filename,
                upload_timestamp=datetime.utcnow()
            )
            session.add(document)
            session.commit()
            logger.debug(f"Saved document record for user {user_id}: {filename}")
    except SQLAlchemyError as e:
        logger.error(f"Error saving document record: {str(e)}")
        raise Exception(f"Failed to save document record: {str(e)}")

def get_user_documents(user_id: str):
    """Get all documents uploaded by a specific user
    
    Args:
        user_id (str): The user ID
        
    Returns:
        list: List of document records as dictionaries
    """
    try:
        with get_db_session() as session:
            documents = session.query(UserDocument)\
                .filter(UserDocument.user_id == user_id)\
                .order_by(UserDocument.upload_timestamp.desc())\
                .all()
            
            # Convert to list of dictionaries for compatibility
            result = []
            for doc in documents:
                result.append({
                    'id': doc.id,
                    'user_id': doc.user_id,
                    'doc_id': doc.doc_id,
                    'filename': doc.filename,
                    'upload_timestamp': doc.upload_timestamp.isoformat()
                })
            
            return result
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving user documents: {str(e)}")
        return []

# Initialize database on import
if __name__ == "__main__":
    # Test the database connection
    if db_manager.test_connection():
        print("Database connection successful!")
    else:
        print("Database connection failed!")
