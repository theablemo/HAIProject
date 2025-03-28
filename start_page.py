import streamlit as st
from streamlit_mic_recorder import mic_recorder
import plotly.express as px
from pipeline import Pipeline
from conversation import Conversation, Solution, Insight
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, RTCConfiguration
import av
import time
import numpy as np
import tempfile
import soundfile as sf
import logging
from AzureSTT import AzureSpeechToText
import os
import pydub
from twilio.rest import Client
import queue
import threading
from concurrent.futures import ThreadPoolExecutor


# This code is based on https://github.com/whitphx/streamlit-webrtc/blob/c1fe3c783c9e8042ce0c95d789e833233fd82e74/sample_utils/turn.py
@st.cache_data  # type: ignore
def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """

    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        # logger.warning(
        #     "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        # )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token.ice_servers

# Set up logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# Dummy solutions list for testing
SOLUTIONS = [
    Solution(
        title="MECHANICAL BACK PAIN",
        subtitle="Due to the nature of the patient's heavy work"
    ),
    Solution(
        title="POSTURAL ISSUES",
        subtitle="Resulting from prolonged sitting"
    ),
]

# Problem description
PROBLEM = "Lower back spasms occurring two to three times a week"

# add dummy insights
INSIGHTS = [
    Insight(
        text="Patient reports increased pain after heavy lifting.",
        sources=["Occupational Health Guidelines, Canada"]
    ),
    Insight(
        text="Symptoms improve with rest and stretching.",
        sources=["Physiotherapy Journal"]
    ),
]

class NewAudioProcessor:
    def __init__(self) -> None:
        self.last_processed = time.time()
        self.transcription_text = ""  # Store transcription text locally
        self.new_transcription_text_added = False
        # NEW
        self.sound_chunk = pydub.AudioSegment.empty()
    
    def recv(self, frame):
        try:
            sound = pydub.AudioSegment(
                    data=frame.to_ndarray().tobytes(),
                    sample_width=frame.format.bytes,
                    frame_rate=frame.sample_rate,
                    channels=len(frame.layout.channels),
                )
            self.sound_chunk += sound
            
            current_time = time.time()
            # logger.info(f"Current time: {current_time}, Last processed: {self.last_processed}")
            if current_time - self.last_processed >= 5:
                # logger.info("Processing 5-second audio chunk")
                
                try:
                    if len(self.sound_chunk) > 0:
                        st.info("Writing wav to disk")
                        self.sound_chunk.export("temp_audio.wav", format="wav")
                    
                    transcription = AzureSpeechToText.transcribe(audio_path="temp_audio.wav")
                    # logger.info(f"Transcription result: {transcription}")
                    
                    self.transcription_text += transcription + "\n"
                    self.new_transcription_text_added = True
                except Exception as e:
                    # logger.error(f"Error processing audio chunk: {e}")
                    pass
                finally:
                    self.sound_chunk = pydub.AudioSegment.empty()
                    self.last_processed = current_time
            return frame
    
        except Exception as e:
            # logger.error(f"Error in recv: {e}")
            pass
            return frame

class AudioProcessor:
    def __init__(self) -> None:
        self.audio_buffer = []
        self.last_processed = time.time()
        self.transcription_text = ""  # Store transcription text locally
        # NEW
        self.sound_chunk = pydub.AudioSegment.empty()
        # logger.info("AudioProcessor initialized")
        
    def recv(self, frame):
        try:
            # Get audio samples from the frame and ensure it's the right shape
            samples = frame.to_ndarray().reshape(-1)  # Flatten to 1D array
            # NEW
            sound = pydub.AudioSegment(
                    data=frame.to_ndarray().tobytes(),
                    sample_width=frame.format.bytes,
                    frame_rate=frame.sample_rate,
                    channels=len(frame.layout.channels),
                )
            self.sound_chunk += sound
            
            # logger.debug(f"Received audio frame: shape={samples.shape}, dtype={samples.dtype}")
            
            # Add frame to buffer
            self.audio_buffer.append(samples)
            current_time = time.time()
            
            # Process every 10 seconds
            if current_time - self.last_processed >= 5:
                # logger.info("Processing 5-second audio chunk")
                
                try:
                    # Concatenate audio data
                    audio_concat = np.concatenate(self.audio_buffer)
                    
                    # Keep as int16 for WAV file
                    if audio_concat.dtype != np.int16:
                        audio_concat = audio_concat.astype(np.int16)
                    
                    # Save audio data to temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        # Save as WAV file with specific parameters
                        sf.write(
                            temp_file.name,
                            audio_concat,
                            samplerate=48000,  # WebRTC typically uses 48kHz
                            format='WAV',
                            subtype='PCM_16'  # 16-bit PCM
                        )
                        # logger.debug(f"Saved audio to temporary file: {temp_file.name}")
                        
                        # Transcribe audio
                        transcription = AzureSpeechToText.transcribe(audio_path=temp_file.name)
                        # logger.info(f"Transcription result: {transcription}")
                        
                        # Add transcription to local buffer if not empty
                        if transcription.strip():
                            self.transcription_text += transcription + "\n"
                            # Update session state through a callback
                            if hasattr(st, 'session_state'):
                                # st.session_state.transcription_text = self.transcription_text
                                st.session_state['transcription_text'] = self.transcription_text
                            # logger.info("Transcription added successfully")
                        
                        # Clean up temporary file
                        # try:
                        #     os.unlink(temp_file.name)
                        # except Exception as e:
                        #     logger.error(f"Error deleting temporary file: {e}")
                
                except Exception as e:
                    # logger.error(f"Error processing audio chunk: {e}")
                    # logger.error(f"Audio shape: {audio_concat.shape}, dtype: {audio_concat.dtype}, max value: {np.max(np.abs(audio_concat))}")
                    pass
                
                finally:
                    # Clear buffer and reset timer regardless of success/failure
                    self.audio_buffer = []
                    self.last_processed = current_time
            
            return frame
        except Exception as e:
            # logger.error(f"Error in recv: {e}")
            pass
        return frame

def transcribe_here(audio_path, transcription_text_in_expander):
    # st.info("Transcribing audio...")
    # logger.info(f"Transcribing audio from {audio_path}")
    transcription = AzureSpeechToText.transcribe(audio_path=audio_path)
    # logger.info(f"Transcription: {transcription}")
    # st.info(f"Transcription: {transcription}")
    # st.session_state.transcription_text += transcription
    # transcription_queue.put(transcription)    
    # get the text that is already in the expander
    text = transcription_text_in_expander.text
    transcription_text_in_expander.text(text + transcription)

class TranscriptionThread(threading.Thread):
    def __init__(self, audio_path):
        super().__init__()
        self.audio_path = audio_path
        self.transcription = None

    def run(self):
        transcription = AzureSpeechToText.transcribe(audio_path=self.audio_path)
        self.transcription = transcription

def main():
    transcription_text_in_expander = None
    transcription_thread = None
    # transcription_queue = queue.Queue()
    # Initialize ALL session state variables at the very beginning
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.started = False
        st.session_state.conversation = None  # Will be initialized later

    # Always ensure these variables are initialized
    if 'transcription_text' not in st.session_state:
        # st.session_state.transcription_text = ""
        st.session_state['transcription_text'] = ""

    if 'azure_stt' not in st.session_state:
        st.session_state.azure_stt = AzureSpeechToText()
    
    # Initialize conversation if not already done
    if st.session_state.conversation is None:
        st.session_state.conversation = Conversation(
            problem_text=PROBLEM,
            solutions=SOLUTIONS,
            insights=INSIGHTS,
            background_info={},
            chat_history=[]
        )
            
    st.set_page_config(layout="wide")
    st.markdown(
        """
        <style>
        div.stButton {
            display: flex;
            justify-content: center;
        }
        .divider {
            border-left: 2px solid gray;
            height: 100%;
            position: absolute;
            left: 50%;
            top: 0;
        }
        .column-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: relative;
        }
        /* Hide WebRTC control elements */
        .streamlit-expanderHeader {
            display: none;
        }
        div[data-testid="stVerticalBlock"] > div:has(button:contains("Stop")) {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initial page with centered "Start" button
    if not st.session_state.started:
        st.markdown("<h1 style='text-align: center;'>Welcome</h1>", unsafe_allow_html=True)
        _, col2, __ = st.columns([1, 2, 1])
        with col2:
            if st.button("Start", key="start"):
                st.session_state.started = True
                st.rerun()
    else:
        # Main UI layout
        # Sidebar with Solutions/Decisions
        with st.sidebar:
            st.markdown("<h2 style='text-align: center;'>Solutions/Decisions</h2>", unsafe_allow_html=True)
            st.markdown("---")
            for solution in st.session_state.conversation.solutions:
                st.markdown(f"**{solution.title}**  \n{solution.subtitle}")
            with st.expander("📃 Transcript", expanded=True):
                # Display transcription text from session state
                # st.write(st.session_state.transcription_text)
                transcription_text_in_expander = st.empty()

        main_content, right_sidebar = st.columns([4, 1], gap="small")
        with right_sidebar:
            # Finish Session button
            if st.button("Finish Session"):
                st.session_state.started = False
                st.rerun()
            st.markdown("---")
            st.markdown("<strong><ins>Background Information</ins></strong>", unsafe_allow_html=True)
            for key, value in st.session_state.conversation.background_info.items():
                st.markdown(f"**{key}:** {value}")

        with main_content:
            # Heading
            st.markdown(f"<h2><ins>Problem Discussion</ins></h2> \n<h3>{st.session_state.conversation.problem_text}</h3>", unsafe_allow_html=True)

            # Display insights in tabs
            if st.session_state.conversation.insights:
                tabs = st.tabs([f"Insight {i+1}" for i in range(len(st.session_state.conversation.insights))])
                for tab, insight in zip(tabs, st.session_state.conversation.insights):
                    with tab:
                        st.markdown(f"**Insights:** {insight.text}")
                        st.markdown(f"**Sources:** {', '.join(insight.sources)}")
                        if insight.vega_lite_spec:
                            st.vega_lite_chart(insight.vega_lite_spec, use_container_width=True)

        st.write("**Voice Recording Status:** Active")
        
        # Create a container for the WebRTC component to better control its visibility
        if "audio_buffer" not in st.session_state:
            st.session_state["audio_buffer"] = pydub.AudioSegment.empty()
        webrtc_container = st.container()
        with webrtc_container:
            current_time = time.time()
            webrtc_ctx = webrtc_streamer(
                key="speech-to-text",
                mode=WebRtcMode.SENDONLY,
                audio_receiver_size=1024,
                rtc_configuration={"iceServers": get_ice_servers()},
                media_stream_constraints={"video": False, "audio": True},
                desired_playing_state=True,
            )
            while True:
                # transcription_text_in_expander.text(" ".join(list(transcription_queue.queue)))
                if transcription_thread is not None and transcription_thread.transcription is not None:
                    # transcription_thread.join()
                    st.session_state.transcription_text += transcription_thread.transcription
                    transcription_text_in_expander.text(st.session_state.transcription_text)
                    transcription_thread = None
                if webrtc_ctx.audio_receiver and webrtc_ctx.state.playing:
                    # st.success("Audio streaming is active")
                    sound_chunk = pydub.AudioSegment.empty()
                    try:
                        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                    except queue.Empty:
                        time.sleep(0.1)
                        continue  
                    
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
                    
                    if time.time() - current_time >= 20:
                        audio_buffer = st.session_state["audio_buffer"]
                        if len(audio_buffer) > 0:
                            # st.info("Writing wav to disk")
                            audio_buffer.export("temp_audio.wav", format="wav")
                            # transcription = AzureSpeechToText.transcribe(audio_path="temp_audio.wav")
                            # st.session_state.transcription_text += transcription
                            # transcription_text_in_expander.text(st.session_state.transcription_text)
                            # thread = threading.Thread(target=transcribe_here, args=("temp_audio.wav", transcription_text_in_expander))
                            # thread.start()
                            transcription_thread = TranscriptionThread("temp_audio.wav")
                            transcription_thread.start()
                            current_time = time.time()
                            # st.success("Transcription added successfully")
                            st.session_state["audio_buffer"] = pydub.AudioSegment.empty()
                            # st.rerun()
            
            
        # with webrtc_container:
        #     # WebRTC component with automatic start
        #     RTC_CONFIGURATION = RTCConfiguration(
        #         {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        #     )
            
        #     webrtc_ctx = webrtc_streamer(
        #         key="audio-recorder",
        #         mode=WebRtcMode.SENDONLY,
        #         audio_receiver_size=256,
        #         rtc_configuration=RTC_CONFIGURATION,
        #         media_stream_constraints={
        #             "video": False,
        #             "audio": True,
        #         },
        #         # async_processing=True,
        #         audio_processor_factory=NewAudioProcessor,
        #         desired_playing_state=True,
        #         #don't show the button
        #         # video_html_attrs={"style": "display: none;"},
        #     )

        #     if webrtc_ctx.state.playing:
        #         st.success("Audio streaming is active")
        #         # Check for new transcriptions
        #         # if hasattr(webrtc_ctx, 'audio_processor') and webrtc_ctx.audio_processor.new_transcription_text_added:
        #         #     st.session_state.transcription_text = webrtc_ctx.audio_processor.transcription_text
        #         #     webrtc_ctx.audio_processor.new_transcription_text_added = False
        #         #     st.rerun()
        #     else:
        #         st.warning("Audio streaming is not active. Please check your microphone settings.")
            
        #     audio_container = st.container()
        #     with audio_container:
        #         st.write("Waiting for transcription...")
        #         time.sleep(1)
        #         while True:
        #             time.sleep(5)
        #             if webrtc_ctx.audio_processor.new_transcription_text_added:
        #                 st.session_state.transcription_text += webrtc_ctx.audio_processor.transcription_text
        #                 st.rerun()   
                        
# Audio recording section
# st.markdown("### Record Audio")
# audio = mic_recorder(start_prompt="Record 15s", stop_prompt="Stop", key="recorder")
# if audio:
#     st.write("Processing audio...")
#     # Here you would call your function with audio['bytes'], e.g., process_audio(audio['bytes'])
#     # st.audio(audio['bytes'])  # Playback the recorded audio as feedback
#     with open("temp_audio.wav", "wb") as f:
#         f.write(audio['bytes'])
        
#     # Process audio and update conversation
#     try:
#         updated_state = st.session_state.pipeline.process_audio(audio['bytes'])
#         st.session_state.conversation = updated_state
#         # st.rerun()
#         st.success("Audio processed successfully")
#     except Exception as e:
#         st.error(f"Error processing audio: {e}")
#     finally:
#         import os
#         if os.path.exists("temp_audio.wav"):
#             os.remove("temp_audio.wav")
        

if __name__ == "__main__":
    main()

