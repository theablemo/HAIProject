from file_embedding_manager import FileEmbeddingManager

def main():
    embedding_manager = FileEmbeddingManager()
    
    # test find similar
    results = embedding_manager.find_similar("I have a back pain", top_k=20)
    for result in results:
        print(result[1])
        print(result[2])
        print(result[3])
        print("--------------------------------")
    

if __name__ == "__main__":
    main()
    