# import os
# import time
# from dotenv import load_dotenv
# from langchain_core.documents.base import Blob
# from langchain_community.document_loaders.parsers.audio import AzureOpenAIWhisperParser

# class AzureSpeechToText:
#     def __init__(self):
#         load_dotenv()
#         """Initialize Azure Speech-to-Text service."""
                
#         # Initialize Whisper parser
#         self.parser = AzureOpenAIWhisperParser(
#             deployment_name="whisper",
#             api_key=os.getenv("AZURE_OPENAI_STT_KEY"),
#             api_version="2024-02-01",
#             azure_endpoint=os.getenv("AZURE_OPENAI_STT_ENDPOINT")
#         )
    
#     def transcribe(self, audio_path: str) -> str:
#         """
#         Transcribe audio file using Azure OpenAI's Whisper model.

#         Args:
#             audio_path (str): Path to the audio file

#         Returns:
#             str: Transcribed text
#         """
#         start_time = time.time()
        
#         # Parse audio file using Whisper
#         blob = Blob(path=audio_path)
#         result = self.parser.parse(blob)
        
#         end_time = time.time()
#         inference_time = end_time - start_time
#         print(f"Inference time: {inference_time:.2f} seconds")
        
#         return result

import os
import time
from openai import AzureOpenAI
from dotenv import load_dotenv

class AzureSpeechToText:
    def __init__(self):
        load_dotenv()
        """Initialize Azure Speech-to-Text service."""
        # Get Azure credentials from environment variables
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_STT_KEY"),  
            api_version="2024-02-01",
            azure_endpoint = os.getenv("AZURE_OPENAI_STT_ENDPOINT")
        )
        self.deployment_id = "whisper" #This will correspond to the custom name you chose for your deployment when you deployed a model."
    
    @staticmethod
    def transcribe(audio_path: str) -> str:
        print("IN AZURE STT")
        load_dotenv()
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_STT_KEY"),  
            api_version="2024-02-01",
            azure_endpoint = os.getenv("AZURE_OPENAI_STT_ENDPOINT")
        )
        deployment_id = "whisper"
        
        start_time = time.time()
        result = client.audio.transcriptions.create(
            file=open(audio_path, "rb"),            
            model=deployment_id
        )
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.2f} seconds")
        return result.text