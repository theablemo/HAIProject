import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import platform
import subprocess
import sys
import time


class SpeechToText:
    def __init__(self):
        self.pipe = self._initialize_model()

    def _check_ffmpeg(self):
        try:
            subprocess.run(
                ["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return True
        except FileNotFoundError:
            print("Error: ffmpeg is not installed. Please install it first:")
            print("On Mac: brew install ffmpeg")
            print("On Linux: sudo apt-get install ffmpeg")
            print("On Windows: choco install ffmpeg")
            sys.exit(1)

    def _get_device(self):
        if platform.processor() == "arm":  # Check for Apple Silicon
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
            except (AttributeError, ImportError):
                pass
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _initialize_model(self):
        # Check for ffmpeg before proceeding
        self._check_ffmpeg()

        device = self._get_device()
        print(f"Using device: {device}")
        torch_dtype = (
            torch.float16
            if (torch.cuda.is_available() or device == "mps")
            else torch.float32
        )

        model_id = "openai/whisper-large-v3-turbo"

        # Model configuration with optimizations
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        # Set the language to English in the model's generation config
        model.generation_config.language = "<|en|>"
        model.generation_config.task = "transcribe"

        if device != "cpu":
            model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            return_timestamps=True,
            batch_size=4,
            chunk_length_s=30,
            stride_length_s=5,
        )

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text.

        Args:
            audio_path (str): Path to the audio file to transcribe

        Returns:
            str: The transcribed text
        """
        start_time = time.time()
        with torch.inference_mode():
            result = self.pipe(audio_path)
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.2f} seconds")
        return result["text"]
