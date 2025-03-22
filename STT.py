import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import platform
import subprocess
import sys


def check_ffmpeg():
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


def get_device():
    if platform.processor() == "arm":  # Check for Apple Silicon
        try:
            import torch.backends.mps

            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def initialize_model():
    # Check for ffmpeg before proceeding
    check_ffmpeg()

    device = get_device()
    torch_dtype = (
        torch.float16
        if (torch.cuda.is_available() or device == "mps")
        else torch.float32
    )

    model_id = "openai/whisper-large-v3"

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


# Initialize the pipeline once when the module is imported
pipe = initialize_model()


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file to text.

    Args:
        audio_path (str): Path to the audio file to transcribe

    Returns:
        str: The transcribed text
    """
    with torch.inference_mode():
        result = pipe(audio_path)
    return result["text"]
