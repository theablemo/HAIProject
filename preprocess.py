from file_embedding_manager import FileEmbeddingManager
import os
from tqdm import tqdm

def process_text_files(embedding_manager: FileEmbeddingManager, data_dir: str):
    """Process text files from the specified directory."""
    folders = os.listdir(data_dir)
    if ".DS_Store" in folders:
        folders.remove(".DS_Store")
    
    for folder in tqdm(folders, desc="Processing text folders"):
        files_dir = os.path.join(data_dir, folder)
        files = os.listdir(files_dir)
        files = [file for file in files if file.endswith(".txt")]
        files_paths = [os.path.join(files_dir, file) for file in files]
        embedding_manager.process_text_files(files_paths)


def process_csv_files(embedding_manager: FileEmbeddingManager, base_dir: str):
    """Process CSV files and their metadata from the specified directory structure."""
    # Get all chunk folders
    folders = [d for d in os.listdir(base_dir) if d.startswith("chunk_")]
    if ".DS_Store" in folders:
        folders.remove(".DS_Store")
    
    for folder in tqdm(folders, desc="Processing CSV chunks"):
        chunk_path = os.path.join(base_dir, folder)
        
        # Get all dataset folders in the chunk
        dataset_dirs = [d for d in os.listdir(chunk_path) if os.path.isdir(os.path.join(chunk_path, d))]
        csv_files = [os.path.join(chunk_path, d, f) for d in dataset_dirs for f in os.listdir(os.path.join(chunk_path, d)) if f.endswith('.csv')]
        print(csv_files)
        if csv_files:
            embedding_manager.process_text_files(csv_files)
        


if __name__ == "__main__":
    embedding_manager = FileEmbeddingManager()
    
    # Process text files
    text_data_dir = "health_data/canada_data/text_data"
    print("Processing text files...")
    process_text_files(embedding_manager, text_data_dir)
    
    # Process CSV files
    csv_data_dir = "health_data/canada_data/csv_datasets_health"
    print("\nProcessing CSV files...")
    process_csv_files(embedding_manager, csv_data_dir)
    
    