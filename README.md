# LangGraph RAG System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system using LangGraph for orchestrating the workflow. The system provides a flexible document retrieval and question-answering capability with support for multiple document collections and intelligent query routing, featuring both CLI and web interface options through Chainlit.

## 🌟 Key Features

- Document ingestion from multiple file formats (PDF, TXT)
- Intelligent query routing between Table of Contents (TOC) and RAG workflows
- Support for multiple document collections
- Flexible embedding model support (OpenAI, Ollama)
- Vector store persistence for efficient document retrieval
- Workflow visualization using Mermaid diagrams
- Interactive web interface using Chainlit
- Real-time streaming of responses

## 🏗️ Architecture

The system consists of two main components:

### 1. Document Retriever

- Handles document ingestion and storage
- Supports multiple file formats
- Manages vector embeddings and similarity search
- Provides collection-based document organization

### 2. RAG Agent System

- Orchestrates the RAG workflow using LangGraph
- Implements intelligent query routing
- Generates responses using retrieved documents
- Provides table of contents functionality
- Supports both CLI and web-based interactions through Chainlit

The workflow follows this process:

1. Query routing (TOC vs RAG)
2. Collection determination (for RAG queries)
3. Document retrieval
4. Response generation

## 🚀 Getting Started

### Prerequisites

```bash
pip install langchain langchain-community langchain-openai langchain-ollama langgraph chromadb python-dotenv pydantic chainlit
```

### Environment Setup

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
```

### Basic Usage

#### CLI Mode

```python
from document_retriever import DocumentRetriever
from rag_agent_system import RAGAgentSystem
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Initialize with Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")
chat_model = ChatOllama(model="mistral")

# Create retriever and system
retriever = DocumentRetriever(embeddings)
rag_system = RAGAgentSystem(retriever, chat_model)

# Add documents
retriever.add_documents_in_directory('./documents/biology', 'biology_docs')

# Query the system
response = rag_system.query("What is DNA?")
print(response)
```

#### Web Interface Mode

Run the system with Chainlit interface:

```bash
chainlit run rag_agent_system.py
```

This will start a web server where you can interact with the RAG system through a user-friendly interface with features like:

- Real-time response streaming
- Session management
- Interactive chat history
- Progress indicators for document retrieval and processing

## 📁 Project Structure

```
.
├── document_retriever.py    # Document ingestion and retrieval
├── rag_agent_system.py      # Main RAG system implementation
├── vectorstore/            # Directory for storing document embeddings
└── documents/             # Directory for source documents
```

## 🛠️ Customization

### Using Different Embedding Models

The system supports various embedding models:

```python
# Using OpenAI embeddings
from langchain_openai import OpenAIEmbeddings
retriever = DocumentRetriever(OpenAIEmbeddings())

# Using Ollama embeddings
from langchain_ollama import OllamaEmbeddings
retriever = DocumentRetriever(OllamaEmbeddings(model="nomic-embed-text"))
```

### Adding New Document Types

Extend the `_get_loader_for_file` method in `DocumentRetriever` to support additional file types:

```python
def _get_loader_for_file(self, file_path: str) -> BaseLoader:
    file_extension = os.path.splitext(file_path)[1].lower()
    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': DocxLoader,  # Add new loader
    }
    # ... rest of the method
```

### Customizing the Web Interface

You can customize the Chainlit interface by modifying the callback handler in `rag_agent_system.py` or by creating a `chainlit.md` file for custom welcome messages and documentation.

## 📊 Workflow Visualization

The system can generate a visual representation of the workflow:

```python
rag_system.draw_graph_diagram()  # Generates workflow.png
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
