from typing import List, Optional, Dict, Any
from datetime import datetime
import os
from dotenv import load_dotenv

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class FileEmbeddingManager:
    def __init__(
        self,
        collection_name: str = "embeddings",
        persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the EmbeddingManager with ChromaDB collection and Google Generative AI embeddings.

        Args:
            collection_name (str): Name of the ChromaDB collection
            persist_directory (str): Directory to persist ChromaDB data
            chunk_size (int): Size of text chunks for splitting documents
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        # Load environment variables
        load_dotenv()

        # Initialize Google Generative AI embeddings
        # self.embeddings = GoogleGenerativeAIEmbeddings(
        #     model="models/gemini-embedding-exp-03-07",
        #     google_api_key=os.getenv("GOOGLE_API_KEY"),
        # )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="Linq-AI-Research/Linq-Embed-Mistral"
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        # Initialize ChromaDB with Langchain
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

    def process_text_files(
        self,
        file_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Process multiple text files and save their chunks as embeddings.

        Args:
            file_paths (List[str]): List of paths to the text files
            metadata (Dict[str, Any], optional): Additional metadata to store with the embeddings

        Returns:
            List[str]: List of embedding IDs for the chunks
        """
        embedding_ids = []

        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Load the text from the file
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            # Split the text into chunks
            chunks = self.text_splitter.split_text(text)

            # Prepare metadata
            if metadata is None:
                metadata = {}
            metadata.update(
                {
                    "source_file": file_path,
                    "created_at": datetime.now().isoformat(),
                    "total_chunks": len(chunks),
                }
            )

            # Save each chunk as an embedding
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                embedding_id = self.save_embedding(chunk, chunk_metadata)
                embedding_ids.append(embedding_id)

        return embedding_ids

    def save_embedding(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a text and its embedding to ChromaDB.

        Args:
            text (str): The text to save
            metadata (Dict[str, Any], optional): Additional metadata to store
            embedding (np.ndarray, optional): The embedding to save. If None, will create one.

        Returns:
            str: The ID of the saved embedding
        """
        # Generate a unique ID
        # embedding_id = f"doc_{datetime.now().timestamp()}"

        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata["created_at"] = datetime.now().isoformat()

        # Create a Document object
        doc = Document(page_content=text, metadata=metadata)

        # Add to ChromaDB using Langchain's interface
        embedding_id = self.vectorstore.add_documents(
            [doc],
            #    ids=[embedding_id],
        )

        return embedding_id[0]

    def find_similar(
        self, query_text: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None
    ) -> List[tuple[str, str, float, Dict[str, Any]]]:
        """
        Find the most similar texts to the query text.

        Args:
            query_text (str): The text to find similar embeddings for
            top_k (int): Number of similar texts to return
            where (Dict[str, Any], optional): Filter conditions for the search

        Returns:
            List[tuple[str, str, float, Dict[str, Any]]]: List of (id, text, similarity_score, metadata) tuples
        """
        # Use Langchain's similarity search with metadata
        results = self.vectorstore.similarity_search_with_score(
            query_text, k=top_k, filter=where
        )

        similarities = []
        for doc, score in results:
            # Extract ID from metadata if available, otherwise generate one
            doc_id = doc.metadata.get("id", f"doc_{datetime.now().timestamp()}")
            similarities.append((doc_id, doc.page_content, float(score), doc.metadata))

        return similarities

    def get_file_chunks(self, file_path: str) -> List[tuple[str, str, Dict[str, Any]]]:
        """
        Retrieve all chunks of a specific file.

        Args:
            file_path (str): Path to the text file

        Returns:
            List[tuple[str, str, Dict[str, Any]]]: List of (embedding_id, text, metadata) tuples
        """
        # Query ChromaDB for all chunks from this file
        results = self.vectorstore.similarity_search_with_score(
            "",  # Empty query to get all documents
            k=1000,  # Adjust based on your needs
            filter={"source_file": file_path},
        )

        chunks = []
        for doc, _ in results:
            doc_id = doc.metadata.get("id", f"doc_{datetime.now().timestamp()}")
            chunks.append((doc_id, doc.page_content, doc.metadata))

        # Sort chunks by their index
        chunks.sort(key=lambda x: x[2].get("chunk_index", 0))
        return chunks
