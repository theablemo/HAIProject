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
    
    def transcribe(self, audio_path: str) -> str:
        start_time = time.time()
        result = self.client.audio.transcriptions.create(
            file=open(audio_path, "rb"),            
            model=self.deployment_id
        )
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.2f} seconds")
        return result.text