# RAG Chatbot with Document Processing

A production-ready Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents, process them with OCR, and ask questions about their content using advanced AI.

## Features

- **Document Processing**: Upload and process PDF, TIFF, JPG, PNG, and text files
- **OCR Integration**: Extract text from images and scanned documents using Tesseract
- **Vector Database**: Store and query document embeddings using Qdrant
- **LLM Integration**: Generate responses using Llama 3 via Ollama
- **User Authentication**: AWS Cognito integration with local fallback
- **Database Support**: PostgreSQL (AWS RDS) with SQLite fallback
- **Multi-User Support**: Each user has isolated document collections and chat history
- **Modern UI**: Clean, responsive interface built with Gradio

## Project Structure

```
src/
├── simple_app.py           # Main application entry point
├── database.py             # Database management (PostgreSQL/SQLite)
├── document_processor.py   # Document upload, OCR, and vector processing
├── chat_history.py         # Chat message and document record management
├── aws_cognito.py          # AWS Cognito authentication
└── [other modules]         # Additional processing modules
```

## Prerequisites

- **Python 3.9+**
- **Qdrant Vector Database**: Running locally or in cloud
- **Ollama with Llama 3**: For LLM responses
- **Tesseract OCR**: For image text extraction
- **PostgreSQL** (optional): For production database

## Quick Start

### 1. Environment Setup

```bash
# Clone and navigate to project
git clone <your-repo-url>
cd "RAG Chatbot Project"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# For local development, you can use SQLite by setting:
# DB_TYPE=sqlite
```

### 3. Start Services

```bash
# Start Qdrant (Docker)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Start Ollama with Llama 3
ollama pull llama3
ollama serve
```

### 4. Run Application

```bash
python src/simple_app.py
```

Open http://127.0.0.1:7860 in your browser.

## Configuration Options

### Database Configuration

**Local Development (SQLite):**
```env
DB_TYPE=sqlite
```

**Production (PostgreSQL):**
```env
DB_HOST=your-rds-endpoint
DB_NAME=postgres
DB_USER=your-username
DB_PASSWORD=your-password
DB_SSL_MODE=require
```

### AWS Cognito (Optional)

```env
AWS_REGION=us-east-2
COGNITO_USER_POOL_ID=your-pool-id
COGNITO_APP_CLIENT_ID=your-client-id
COGNITO_APP_CLIENT_SECRET=your-client-secret
```

## Usage

1. **Login**: Use any username for local development
2. **Upload Documents**: Go to "Upload Documents" tab
3. **Process Files**: Upload PDFs, images, or text files
4. **Chat**: Ask questions about your uploaded documents
5. **History**: View past conversations in "Chat History" tab

## Production Deployment

### AWS Setup

1. **Database**: Set up AWS RDS PostgreSQL instance
2. **Authentication**: Configure AWS Cognito User Pool
3. **Vector DB**: Deploy Qdrant on EC2 or use managed service
4. **Application**: Deploy to EC2, ECS, or serverless

### Security Considerations

- Never commit `.env` files to version control
- Use AWS IAM roles for production authentication
- Enable SSL/TLS for all database connections
- Regularly rotate API keys and passwords

## Troubleshooting

### Common Issues

**Database Connection Failed:**
- Check PostgreSQL credentials and network access
- Verify RDS security groups allow connections
- Try SQLite fallback: `DB_TYPE=sqlite`

**Qdrant Connection Failed:**
- Ensure Qdrant is running on port 6333
- Check firewall settings

**OCR Not Working:**
- Install Tesseract: `brew install tesseract` (macOS) or `apt-get install tesseract-ocr` (Ubuntu)

## Development

### Adding New Features

1. Create feature branch
2. Update relevant modules in `src/`
3. Test with both SQLite and PostgreSQL
4. Update documentation

### Testing

```bash
# Run basic connection tests
python -c "from src.database import db_manager; print('DB OK' if db_manager.test_connection() else 'DB Failed')"
```

## License

MIT License - See LICENSE file for details
