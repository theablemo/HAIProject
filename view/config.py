import streamlit as st
from agent.models import Solution, Insight

# Development mode flag - set to True for testing with sample data
DEV_MODE = False

# Default processing interval in seconds
AUDIO_PROCESSING_INTERVAL = 20

# Initial empty values
SOLUTIONS = (
    [
        Solution(
            title="MECHANICAL BACK PAIN",
            subtitle="Due to the nature of the patient's heavy work",
        ),
        Solution(title="POSTURAL ISSUES", subtitle="Resulting from prolonged sitting"),
        Solution(
            title="START TALKING TO ADD SOLUTIONS",
            subtitle="Solutions will appear here",
        ),
    ]
    if DEV_MODE
    else []
)

# Problem description
PROBLEM = "Lower back spasms occurring two to three times a week" if DEV_MODE else ""

# Insights list
INSIGHTS = (
    [
        Insight(
            text="Patient reports increased pain after heavy lifting.",
            sources=["Occupational Health Guidelines, Canada"],
        ),
        Insight(
            text="Symptoms improve with rest and stretching.",
            sources=["Physiotherapy Journal"],
        ),
        Insight(
            text="START TALKING TO ADD INSIGHTS",
            sources=["Sources will appear here"],
        ),
    ]
    if DEV_MODE
    else []
)


# Session state initialization
def initialize_session_state():
    """Initialize all required session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.started = False
        st.session_state.conversation = None

    if "transcription_text" not in st.session_state:
        st.session_state["transcription_text"] = ""

    if "azure_stt" not in st.session_state:
        from agent.Azure_STT import AzureSpeechToText

        st.session_state.azure_stt = AzureSpeechToText()

    if "countdown" not in st.session_state:
        st.session_state.countdown = AUDIO_PROCESSING_INTERVAL

    if "audio_buffer" not in st.session_state:
        import pydub

        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

    if "transcription_thread" not in st.session_state:
        st.session_state.transcription_thread = None
