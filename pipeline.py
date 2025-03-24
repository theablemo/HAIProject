from AzureSTT import AzureSpeechToText
from STT import SpeechToText

def main():
    # Create an instance
    # stt = SpeechToText()

    # # Use it to transcribe audio
    # transcript = stt.transcribe("Backend/test.m4a")
    # transcript = stt.transcribe("Backend/test.m4a")
    azure_stt = AzureSpeechToText()
    transcript = azure_stt.transcribe("Backend/test.m4a")
    print(transcript)

if __name__ == "__main__":
    main()