from typing import List, Optional, Dict, Any
from datetime import datetime
import os
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_community.document_loaders import CSVLoader
import pandas as pd

from langchain_openai import AzureOpenAIEmbeddings
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
        Initialize the EmbeddingManager with ChromaDB collection and Azure OpenAI embeddings.

        Args:
            collection_name (str): Name of the ChromaDB collection
            persist_directory (str): Directory to persist ChromaDB data
            chunk_size (int): Size of text chunks for splitting documents
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        # Load environment variables
        load_dotenv()

        # Initialize Azure OpenAI embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            chunk_size=chunk_size
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

    def extract_metadata_and_content(self, text: str) -> tuple[Dict[str, Any], str]:
        """
        Extract metadata and content from a text file.

        Args:
            text (str): The full text content of the file

        Returns:
            tuple[Dict[str, Any], str]: A tuple containing (metadata dictionary, content text)
        """
        metadata_lines = []
        content_lines = []
        in_metadata = True
        
        for line in text.split('\n'):
            if line.startswith('-------------------------------'):
                in_metadata = False
                continue
            if in_metadata:
                metadata_lines.append(line)
            else:
                content_lines.append(line)

        # Parse metadata
        file_metadata = {}
        for line in metadata_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key == 'Keywords':
                    # Convert string representation of list to actual list
                    value = eval(value)
                    # Convert list to comma-separated string for ChromaDB compatibility
                    value = ', '.join(value)
                file_metadata[key] = value

        # Join the content lines back together
        content_text = '\n'.join(content_lines)

        return file_metadata, content_text

    def process_csv_with_metadata(
        self,
        csv_path: str,
        metadata_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Process a CSV file's metadata and save it as an embedding.
        Skips files that have already been embedded in the database.

        Args:
            csv_path (str): Path to the CSV file
            metadata_path (str): Path to the metadata text file
            metadata (Dict[str, Any], optional): Additional metadata to store with the embeddings

        Returns:
            List[str]: List of embedding IDs (will be a single ID for the metadata)
        """
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found: {csv_path}")
            return []
        if not os.path.exists(metadata_path):
            print(f"Warning: Metadata file not found: {metadata_path}")
            return []

        # Check if file is already in the database
        existing_chunks = self.vectorstore.similarity_search_with_score(
            "",  # Empty query to get all documents
            k=1,  # We only need to check if any chunks exist
            filter={"source_file": csv_path},
        )
        
        if existing_chunks:
            print(f"Skipping {csv_path} - already embedded in database")
            return []

        # Extract metadata from the metadata file
        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata_text = file.read()
        except Exception as e:
            print(f"Warning: Could not read metadata file {metadata_path}: {str(e)}")
            return []
        
        file_metadata, _ = self.extract_metadata_and_content(metadata_text)
        
        # Read CSV headers
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Warning: Error reading CSV with {encoding} encoding: {str(e)}")
                    continue
            
            if df is None:
                print(f"Warning: Could not read CSV file {csv_path} with any supported encoding")
                csv_headers = []
            else:
                csv_headers = list(df.columns)
        except Exception as e:
            print(f"Warning: Could not read CSV headers from {csv_path}: {str(e)}")
            csv_headers = []
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata.update(file_metadata)
        metadata.update(
            {
                "source_file": csv_path,
                "metadata_file": metadata_path,
                "created_at": datetime.now().isoformat(),
                "file_type": "csv",
                "csv_headers": ', '.join(csv_headers),
            }
        )

        # Create a descriptive text from the metadata for embedding
        metadata_description = f"CSV file. "
        if csv_headers:
            metadata_description += f"Columns: {', '.join(csv_headers)}. "
        if 'Keywords' in metadata:
            metadata_description += f"Keywords: {metadata['Keywords']}. "
        if 'Notes' in metadata:
            metadata_description += f"Notes: {metadata['Notes']}"

        try:
            # Save the metadata as a single embedding
            embedding_id = self.save_embedding(metadata_description, metadata)
            return [embedding_id]
        except Exception as e:
            print(f"Warning: Failed to save embedding for {csv_path}: {str(e)}")
            return []

    def process_text_files(
        self,
        file_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Process multiple text files and save their chunks as embeddings.
        Skips files that have already been embedded in the database.

        Args:
            file_paths (List[str]): List of paths to the text files
            metadata (Dict[str, Any], optional): Additional metadata to store with the embeddings

        Returns:
            List[str]: List of embedding IDs for the chunks
        """
        embedding_ids = []
        
        for file_idx, file_path in enumerate(tqdm(file_paths, desc="Processing files")):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Check if file is already in the database
            existing_chunks = self.vectorstore.similarity_search_with_score(
                "",  # Empty query to get all documents
                k=1,  # We only need to check if any chunks exist
                filter={"source_file": file_path},
            )
            
            if existing_chunks:
                print(f"Skipping {file_path} - already embedded in database")
                continue

            # Check if this is a CSV file with metadata
            if file_path.endswith('.csv'):
                metadata_path = file_path.replace('.csv', '_metadata.txt')
                if os.path.exists(metadata_path):
                    csv_embedding_ids = self.process_csv_with_metadata(
                        file_path,
                        metadata_path,
                        metadata
                    )
                    embedding_ids.extend(csv_embedding_ids)
                    continue

            # Process as regular text file
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            # Extract metadata and content
            file_metadata, content_text = self.extract_metadata_and_content(text)

            # Split the text into chunks
            chunks = self.text_splitter.split_text(content_text)

            # Prepare metadata
            if metadata is None:
                metadata = {}
            metadata.update(file_metadata)
            metadata.update(
                {
                    "source_file": file_path,
                    "created_at": datetime.now().isoformat(),
                    "total_chunks": len(chunks),
                    "file_type": "text",
                }
            )

            # Save each chunk as an embedding
            for i, chunk in enumerate(tqdm(chunks, desc=f"Processing chunks for file {file_idx + 1}/{len(file_paths)}", leave=False)):
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
            doc_id = doc.id
            # doc_id = doc.metadata.get("id", f"doc_{datetime.now().timestamp()}")
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

    def remove_embeddings_by_file_type(self, file_type: str) -> int:
        """
        Remove all embeddings that have a specific file type.

        Args:
            file_type (str): The file type to filter by (e.g., "csv", "text")

        Returns:
            int: Number of embeddings removed
        """
        try:
            # Get the collection
            collection = self.vectorstore._collection
            
            # Delete documents where file_type matches using where clause
            collection.delete(
                where={"file_type": file_type}
            )
            
            print(f"Removed all embeddings with file_type '{file_type}'")
            return 1  # Return 1 to indicate successful deletion
        except Exception as e:
            print(f"Error removing embeddings: {str(e)}")
            return 0
