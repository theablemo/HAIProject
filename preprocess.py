from file_embedding_manager import FileEmbeddingManager
import os

if __name__ == "__main__":
    embedding_manager = FileEmbeddingManager()
    # get the list of all folders in the directory data/canada_data/text_data
    folders = os.listdir("data/canada_data/text_data")
    # remove the .DS_Store file from the list
    folders.remove(".DS_Store")
    for folder in folders:
        files_dir = os.path.join("data/canada_data/text_data", folder)
        files = os.listdir(files_dir)
        embedding_manager.process_text_files(
            [os.path.join(files_dir, file) for file in files]
        )
        break
