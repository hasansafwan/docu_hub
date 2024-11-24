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
- Interactive web interface using Chainlit with real-time streaming
- Session management and conversation history
- Progress tracking for document processing
- Batch processing for large document sets
- Customizable chunk sizes and overlap for document splitting

## 🏗️ Architecture

The system consists of two main components:

### 1. Document Retriever

- Handles document ingestion and storage
- Supports multiple file formats
- Manages vector embeddings and similarity search
- Provides collection-based document organization
- Implements batch processing for memory efficiency
- Supports customizable document chunking strategies

### 2. RAG Agent System

- Orchestrates the RAG workflow using LangGraph
- Implements intelligent query routing
- Generates responses using retrieved documents
- Provides table of contents functionality
- Supports both CLI and web-based interactions through Chainlit
- Features step-by-step progress tracking with emoji indicators

The workflow follows this process:

1. Query routing (TOC vs RAG)
2. Collection determination (for RAG queries)
3. Document retrieval
4. Response generation

## 🚀 Getting Started

### Prerequisites

Install dependencies using requirements.txt:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install langchain langchain-community langchain-openai langchain-ollama langgraph chromadb python-dotenv pydantic chainlit ollama<0.4.0
```

Note: Ollama version must be below 0.4.0 for compatibility.

### Environment Setup

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
```

### Usage Options

#### 1. CLI Mode

```python
from document_retriever import DocumentRetriever
from rag_agent_system import RAGAgentSystem
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Initialize with Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")
chat_model = ChatOllama(model="mistral", temperature=0.0)

# Create retriever and system
retriever = DocumentRetriever(embeddings)
rag_system = RAGAgentSystem(retriever, chat_model)

# Add documents
retriever.add_documents_in_directory('./documents/biology', 'biology_docs')

# Query the system
response = rag_system.query("What is DNA?")
print(response)
```

#### 2. Web Interface Mode (Chainlit)

##### Creating a Chainlit App

1. Create a `chainlit.md` file in your project root to customize the welcome message:

```markdown
# Welcome to RAG Assistant! 👋

This is a Retrieval-Augmented Generation (RAG) system that can help answer your questions based on the available document collections.

## Features

- Intelligent query routing
- Table of Contents generation
- Document-based question answering
- Real-time response streaming

## How to use

1. Ask about available topics using questions like "What information is available?"
2. Ask specific questions about the documents in the collections
3. Watch as the system retrieves and processes relevant information in real-time
```

##### Running the Chainlit App

Start the web interface with:

```bash
chainlit run rag_agent_system.py
```

This will:

1. Start a local web server (typically at http://localhost:8000)
2. Open a browser window with the interactive chat interface
3. Enable real-time streaming of responses
4. Provide session management and chat history
5. Show progress indicators for each processing step

## 📁 Project Structure

```
.
├── document_retriever.py    # Document ingestion and retrieval
├── rag_agent_system.py      # Main RAG system implementation
├── chainlit.md             # Chainlit welcome message and documentation
├── requirements.txt        # Project dependencies
├── vectorstore/            # Directory for storing document embeddings
└── documents/             # Directory for source documents
```

## 🛠️ Customization

### Using Different Embedding Models

```python
# Using OpenAI embeddings
from langchain_openai import OpenAIEmbeddings
retriever = DocumentRetriever(OpenAIEmbeddings())

# Using Ollama embeddings
from langchain_ollama import OllamaEmbeddings
retriever = DocumentRetriever(OllamaEmbeddings(model="nomic-embed-text"))
```

### Configuring Document Processing

You can customize document processing parameters:

```python
retriever = DocumentRetriever(
    embedding_model=embeddings,
    vector_store_path='./custom_vectorstore',
    max_batch_size=50  # Adjust batch size for memory management
)

# Custom document splitting
documents = retriever._split_documents(
    documents,
    chunk_size=500,    # Adjust chunk size
    chunk_overlap=50   # Adjust overlap
)
```

### Adding New Document Types

Extend the `_get_loader_for_file` method in `DocumentRetriever`:

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

### Customizing the Chainlit Interface

The system includes built-in step tracking with emojis:

- 📋 Analyzing Query Type
- 🔍 Determining Collection
- 📚 Retrieving Documents
- 💭 Generating Response
- 📑 Generating Table of Contents
- ✨ Formatting Table of Contents

You can customize these in the `create_step` function in `rag_agent_system.py`.

## 📊 Workflow Visualization

Generate a visual representation of the workflow:

```python
rag_system.draw_graph_diagram()  # Generates workflow.png
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
