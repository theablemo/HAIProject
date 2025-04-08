import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import pydub
import time
import os
import queue
import threading
from twilio.rest import Client
from agent.Azure_STT import AzureSpeechToText


# Get ICE servers for WebRTC
@st.cache_data
def get_ice_servers():
    """Get ICE servers for WebRTC, using Twilio if credentials are available."""
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)
    token = client.tokens.create()
    return token.ice_servers


class TranscriptionThread(threading.Thread):
    """Background thread for processing audio transcription."""

    def __init__(self, audio_path, pipeline, conversation):
        super().__init__()
        self.audio_path = audio_path
        self.pipeline = pipeline
        self.conversation = conversation
        self.transcription = None
        self.updated_conversation = None

    def run(self):
        """Execute transcription and conversation update in background."""
        transcription = AzureSpeechToText.transcribe(audio_path=self.audio_path)
        self.transcription = transcription
        if transcription:
            self.updated_conversation = self.pipeline.process_audio(transcription)


def setup_audio_capture(webrtc_container):
    """Set up WebRTC audio streaming and processing."""
    with webrtc_container:
        # Start time to track processing intervals
        current_time = time.time()

        # Initialize WebRTC
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            rtc_configuration={"iceServers": get_ice_servers()},
            media_stream_constraints={"video": False, "audio": True},
            desired_playing_state=True,
        )

        return webrtc_ctx, current_time


def process_audio_frames(webrtc_ctx, current_time):
    """Process audio frames from WebRTC stream and handle transcription.

    Returns:
        tuple: (transcription_thread, updated_current_time, should_continue)
    """
    transcription_thread = None
    should_continue = True

    # Process transcription results if available
    if (
        hasattr(st.session_state, "transcription_thread")
        and st.session_state.transcription_thread is not None
        and st.session_state.transcription_thread.transcription is not None
        and st.session_state.transcription_thread.updated_conversation is not None
    ):

        st.session_state.transcription_text += (
            st.session_state.transcription_thread.transcription
        )
        st.session_state.conversation = (
            st.session_state.transcription_thread.updated_conversation
        )
        st.session_state.transcription_thread = None
        return None, current_time, False  # Signal to rerun the app

    # Collect audio frames
    if webrtc_ctx.audio_receiver and webrtc_ctx.state.playing:
        sound_chunk = pydub.AudioSegment.empty()
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            time.sleep(0.1)
            return None, current_time, True

        # Process audio frames
        for audio_frame in audio_frames:
            sound = pydub.AudioSegment(
                data=audio_frame.to_ndarray().tobytes(),
                sample_width=audio_frame.format.bytes,
                frame_rate=audio_frame.sample_rate,
                channels=len(audio_frame.layout.channels),
            )
            sound_chunk += sound

            if len(sound_chunk) > 0:
                st.session_state["audio_buffer"] += sound_chunk

        # Process accumulated audio every 20 seconds
        if time.time() - current_time >= 20:
            audio_buffer = st.session_state["audio_buffer"]
            if len(audio_buffer) > 0:
                audio_buffer.export("temp_audio.wav", format="wav")
                transcription_thread = TranscriptionThread(
                    "temp_audio.wav",
                    st.session_state.pipeline,
                    st.session_state.conversation,
                )
                transcription_thread.start()
                current_time = time.time()
                st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

    return transcription_thread, current_time, should_continue
