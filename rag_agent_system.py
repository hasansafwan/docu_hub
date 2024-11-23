from typing import Dict, List, Any
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from document_retriever import DocumentRetriever

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl

load_dotenv()

class AgentState(BaseModel):
    """State for the RAG agent system"""
    messages: List[Dict] = Field(default_factory=list)
    collection_name: str = ""
    retrieved_docs: List[str] = Field(default_factory=list)
    next_step: str = "route_query"
    is_toc_request: bool = False
    toc_data: Dict[str, Any] = Field(default_factory=dict)


class RAGAgentSystem:
    def __init__(self, retriever: DocumentRetriever, llm: BaseChatModel = None):
        self.retriever = retriever
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo")
        self.workflow = self._create_workflow()

    def _create_route_prompt(self) -> ChatPromptTemplate:
        """Create prompt for determining if query is TOC-related"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a query router that determines if a user is requesting
             table of contents information. Analyze the query and respond with ONLY 'TOC'
             if the user is asking about:
             - Available topics
             - Document contents overview
             - what information is available
             - System content summary
             - List of topics
             Otherwise, respond with 'RAG'.
             ONLY respond with 'TOC' or 'RAG', nothing else."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def _create_toc_prompt(self) -> ChatPromptTemplate:
        """Create prompt for generating TOC response"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that generates a table of contents
             from document collections. Based on the available topics and documents, create
             a clear, hierarchical overview.
             
             Available collections and their topics:
             {toc_data}
             
             Present the information in a clear, structured format using:
             - Emoji for visual organization (📚, 📂, 📄)
             - Clear hierarchical structure
             - Numbering for topics and subtopics"""),
             MessagesPlaceholder(variable_name="messages"),
        ])
    
    def _route_query(self, state: AgentState) -> AgentState:
        """Determine whether to process as TOC request of regular query"""
        chain = self._create_route_prompt() | self.llm

        last_message = state.messages[-1]["content"]
        response = chain.invoke({
            "messages": [HumanMessage(content=last_message)]
        })

        state.is_toc_request = response.content.strip() == "TOC"
        state.next_step = "generate_toc" if state.is_toc_request else "determine_collection"
        return state
    
    def _generate_toc(self, state: AgentState) -> AgentState:
        """Generate table of contents overview"""
        # Gather TOC data from all collections
        toc_data = {}
        for collection in self.retriever.list_collections():
            # Get sample documents to analyze topics
            docs = self.retriever.retrieve_documents(
                query="overview",
                collection_name=collection,
                top_k=5
            )

            # Extract topics using LLM
            topic_prompt = [
                ("system", """analyze the following document content and identify main topics.
                 Return ONLY a JSON array of topics, nothing else."""),
                 ("user", "\n\n".join([doc.page_content for doc in docs]))
            ]

            topics_response = self.llm.invoke(topic_prompt)
            try:
                topics = json.loads(topics_response.content)
            except:
                topics = ["General Content"]
            
            toc_data[collection] = topics
        
        # Store TOC data in state
        state.toc_data = toc_data

        state.next_step = "format_toc"
        return state
    
    def _format_toc(self, state: AgentState) -> AgentState:
        """Return formatted table of contents"""
        # Generate formatted TOC response
        chain = self._create_toc_prompt() | self.llm

        response = chain.invoke({
            "messages": [HumanMessage(content=state.messages[-1]["content"])],
            "toc_data": json.dumps(state.toc_data, indent=2)
        })

        state.messages.append({
            "role": "assistant",
            "content": response.content
        })

        state.next_step = END
        return state

    def _create_collection_selector_prompt(self) -> ChatPromptTemplate:
        """Create prompt for selecting the appropriate collection"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that determines which document collection to search based on user queries.
             Available collections: {collections}
             
             Anaylze the user's query and determine the most appropriate collection to search.
             Respond with ONLY the collection name, nothing else.
             """),
             MessagesPlaceholder(variable_name="messages"),
        ])
    
    def _create_rag_prompt(self) -> ChatPromptTemplate:
        """Create prompt for the RAG response generation"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on retrieved documents.
             Use the following retrieved documents to answer the user's question:
             {retrieved_docs}
             
             If you can't find the answer in the documents, say so clearly."""),
             MessagesPlaceholder(variable_name="messages"),
        ])
    
    def _determine_collection(self, state: AgentState) -> AgentState:
        """Determine which collection to search based on the query"""
        collections = self.retriever.list_collections()

        # Create the collection selector chain
        chain = self._create_collection_selector_prompt() | self.llm

        # Get the last user message
        last_message = state.messages[-1]["content"]

        # Determine collection
        response = chain.invoke({
            "messages": [HumanMessage(content=last_message)],
            "collections": ", ".join(collections)
        })

        state.collection_name = response.content.strip()
        state.next_step = "retrieve_documents"
        return state
    
    def _retrieve_documents(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents from the selected collection"""
        last_message = state.messages[-1]["content"]

        # Retrieve documents
        docs = self.retriever.retrieve_documents(
            query=last_message,
            collection_name=state.collection_name
        )

        # Extract and store document content
        state.retrieved_docs = [doc.page_content for doc in docs]
        state.next_step = "generate_response"
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate a response based on retrieved documents"""
        # Create the RAG chain
        chain = self._create_rag_prompt() | self.llm

        # Generate response
        response = chain.invoke({
            "messages": [HumanMessage(content=state.messages[-1]["content"])],
            "retrieved_docs": "\n\n".join(state.retrieved_docs)
        })

        # Add response to messages
        state.messages.append({
            "role": "assistant",
            "content": response.content
        })

        state.next_step = END
        return state
    
    def _create_workflow(self) -> CompiledStateGraph:
        """Create the agent workflow graph"""
        # Define workflow nodes
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("generate_toc", self._generate_toc)
        workflow.add_node("format_toc", self._format_toc)
        workflow.add_node("determine_collection", self._determine_collection)
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("generate_response", self._generate_response)

        # Add edges
        workflow.add_edge(START, "route_query")

        # Add conditional edges
        workflow.add_conditional_edges(
            "route_query",
            lambda x: x.next_step,
            {
                "generate_toc": "generate_toc",
                "determine_collection": "determine_collection"
            }
        )
        
        workflow.add_edge("determine_collection", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("generate_toc", "format_toc")
        workflow.add_edge("format_toc", END)

        return workflow.compile()
    
    def query(self, user_input: str) -> str:
        """Process a user query and return a response"""
        # Initialize state
        state = AgentState(
            messages=[{"role": "user", "content": user_input}]
        )

        # Execute workflow
        final_state = self.workflow.invoke(state)

        # Return the assistant's response
        return final_state.get('messages')[-1]["content"]
    
    def draw_graph_diagram(self) -> None:
        """Get the graph diagram of the agent workflow"""
        bytes = self.workflow.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
        with open("workflow.png", "wb") as f:
            f.write(bytes)
    

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize the document retriever
retriever = DocumentRetriever(embeddings)
    
# Initialize the RAG agent system
rag_system = RAGAgentSystem(retriever)

@cl.on_message
async def on_message(msg: cl.Message):
    config = {
        "configurable":
        {
            "thread_id": cl.context.session.id
        }
    }
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")

    # Initialize state
    state = AgentState(
        messages=[{"role": "user", "content": msg.content}]
    )

    for msg, metadata in rag_system.workflow.stream(state, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
        if (
            msg.content and not isinstance(msg, HumanMessage)
            and metadata["langgraph_node"] in ["format_toc", "generate_response"]
        ):
            await final_answer.stream_token(msg.content)
    
    await final_answer.send()

# Example usage
if __name__ == "__main__":
    # ollama embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # mistral chat model
    chat_model = ChatOllama(model="mistral")
    # Initialize the document retriever
    retriever = DocumentRetriever(embeddings)
    
    # Initialize the RAG agent system
    rag_system = RAGAgentSystem(retriever)

    rag_system.draw_graph_diagram()

    # create loop to keep asking for user input
    while True:
        query = input("Enter your query (q or quit to exit): ")
        if query in ["q", "quit"]:
            break
        response = rag_system.query(query)
        print(f"Response: {response}")