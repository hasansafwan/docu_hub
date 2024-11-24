import os
from typing import List
from dotenv import load_dotenv
import math
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
import chromadb

load_dotenv()

class DocumentRetriever:
    def __init__(self, embedding_model=None, vector_store_path='./vectorstore', max_batch_size=100):
        """
        Initialize the document retriever with optional embedding model and vector store path

        Args:
            embedding_model: Embedding model to use (defaults to OpenAI embeddings)
            vector_store_path: Path to store vector database
            max_batch_size: Maximum number of documents to process in a single batch
        """
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.max_batch_size = max_batch_size

        # Create vector store directory if it doesn't exist
        os.makedirs(vector_store_path, exist_ok=True)
    
    def _get_loader_for_file(self, file_path: str) -> BaseLoader:
        """
        Determine the appropriate document loader based on file extension

        Returns:
            Appropriate document loader
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            # Add more loaders here as needed
        }

        loader_class = loaders.get(file_extension)
        if not loader_class:
            raise ValueError(f'Unsupported file type: {file_extension}')
        
        return loader_class(file_path)
    
    def _split_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Split documents into smaller chunks for better embedding and retrieval.
        
        Args:
            documents: List of documents to split
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks

        Returns:
            List of split document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)
    
    def _process_documents_in_batches(self, documents: List[Document], collection_name: str):
        """
        Process documents in batches to avoid memory issues.

        Args:
            documents: List of documents to process
            collection_name: Name of the collection to store documents
        """
        vector_store_path = os.path.join(self.vector_store_path, collection_name)
        
        # Calculate number of batches
        num_batches = math.ceil(len(documents) / self.max_batch_size)
        
        # Create Chroma client
        chroma_client = chromadb.PersistentClient(path=vector_store_path)
        
        # Process first batch to initialize the vector store
        first_batch = documents[:self.max_batch_size]
        vector_store = Chroma.from_documents(
            documents=first_batch,
            embedding=self.embedding_model,
            persist_directory=vector_store_path,
            client=chroma_client,
            collection_name=collection_name
        )
        
        # Process remaining batches
        if num_batches > 1:
            for i in tqdm(range(1, num_batches), desc="Processing document batches"):
                start_idx = i * self.max_batch_size
                end_idx = min((i + 1) * self.max_batch_size, len(documents))
                batch = documents[start_idx:end_idx]
                
                # Add batch to existing vector store
                vector_store.add_documents(documents=batch)
    
    def add_documents(self, file_paths: List[str], collection_name: str = 'default'):
        """
        Add documents to the vector store.

        Args:
            file_paths: List of file paths to add
            collection_name: Name of the collection to store documents
        """
        all_documents = []

        for file_path in file_paths:
            loader = self._get_loader_for_file(file_path)
            documents = loader.load()
            all_documents.extend(documents)
        
        # Split documents into chunks
        split_docs = self._split_documents(all_documents)
        
        # Process documents in batches
        self._process_documents_in_batches(split_docs, collection_name)

    def add_documents_in_directory(self, directory_path: str, collection_name: str = 'default'):
        """
        Add all documents in a directory to the vector store.

        Args:
            directory_path: Path to the directory containing documents
            collection_name: Name of the collection to store documents
        """
        file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path)
                     if os.path.isfile(os.path.join(directory_path, f))]
        self.add_documents(file_paths, collection_name)
    
    def retrieve_documents(self, query: str, collection_name: str = 'default', top_k: int = 5) -> List[Document]:
        """
        Retrieve documents from a specific collection based on a query.

        Args:
            query: Search query
            collection_name: Name of the collection to search
            top_k: Number of documents to retrieve

        Returns:
            List of most relevant documents
        """
        vector_store_path = os.path.join(self.vector_store_path, collection_name)
        chroma_client = chromadb.PersistentClient(path=vector_store_path)
        
        vector_store = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=self.embedding_model
        )

        return vector_store.similarity_search(query, k=top_k)
    
    def list_collections(self) -> List[str]:
        """
        List all available collections in the vector store.

        Returns:
            List of collection names
        """
        return [name for name in os.listdir(self.vector_store_path)
                if os.path.isdir(os.path.join(self.vector_store_path, name))]
    

# Example usage
if __name__ == '__main__':
        # ollama embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    retriever = DocumentRetriever(embeddings)

    # Add documents to different collections
    retriever.add_documents_in_directory('./documents/programming', collection_name='programming_docs')

    # Retrieve documents based on a query
    results = retriever.retrieve_documents("what is C#", collection_name='programming_docs', top_k=5)
    for doc in results:
        print(doc.page_content)