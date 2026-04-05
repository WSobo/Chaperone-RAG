from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chaperone.utils.logger import logger
from typing import List
import os

class RAGMemory:
    def __init__(self, db_dir: str = "data/chroma_db", collection_name: str = "chaperone_docs"):
        self.db_dir = db_dir
        self.collection_name = collection_name
        
        logger.info(f"Initializing RAG Memory (Chroma DB at {self.db_dir})")
        # In a generic offline cluster environment or fast deployment, an all-MiniLM model is robust
        self.embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load or create vector store
        self.vectorstore = Chroma(
            collection_name=self.collection_name, 
            embedding_function=self.embedding_fn,
            persist_directory=self.db_dir
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        
        self.rfdiffusion_docs = ""
        
    def ingest_pdfs(self, folder_path: str = "data/papers"):
        """Ingests all PDFs from a given local directory into Chroma DB."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"Created empty directory {folder_path} for PDFs.")
            return
            
        logger.info(f"Ingesting PDFs from {folder_path} into RAG database...")
        loader = PyPDFDirectoryLoader(folder_path)
        docs = loader.load()
        if not docs:
            logger.info("No PDFs found to ingest.")
            return
            
        chunks = self.text_splitter.split_documents(docs)
        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()
        logger.info(f"Successfully ingested and indexed {len(chunks)} document chunks from PDFs.")

    def ingest_urls(self, urls: List[str]):
        """Ingests content from a list of URLs and adds them to Chroma DB."""
        logger.info(f"Ingesting {len(urls)} URLs into RAG database...")
        loader = WebBaseLoader(web_paths=urls)
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        
        # Keep a short text snippet around just in case the engine explicitly expects text directly
        if chunks:
            self.rfdiffusion_docs = chunks[0].page_content
        
        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()
        logger.info(f"Successfully ingested and indexed {len(chunks)} document chunks from Web.")

    def search_context(self, query: str, k: int = 3) -> str:
        """Retrieves and formats top K context snippets for a query."""
        results = self.vectorstore.similarity_search(query, k=k)
        if not results:
            return "No relevant context found."
            
        context_blocks = [f"Source [{r.metadata.get('source', 'Unknown')}]:\n {r.page_content}" for r in results]
        return "\n\n---\n\n".join(context_blocks)

    def get_docs(self) -> str:
        """Legacy fallback to retrieve the first scraped doc fragment."""
        if not self.rfdiffusion_docs:
            logger.warning("Docs cache empty! Attempting fallback ingest of RFdiffusion docs...")
            self.ingest_urls(["https://raw.githubusercontent.com/RosettaCommons/RFdiffusion/main/README.md"])
            
        return self.rfdiffusion_docs
