import whisper
import os

# Load the Whisper model
model = whisper.load_model("base")


def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio from a file path using Whisper.

    Args:
        file_path (str): Path to the audio file

    Returns:
        str: Transcribed text from the audio file

    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: If there's an error during transcription
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at path: {file_path}")

    try:
        # Transcribe the audio file
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        raise Exception(f"Error during transcription: {str(e)}")
