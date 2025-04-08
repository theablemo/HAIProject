import streamlit as st
import time
import pydub

# Import local modules
from agent.pipeline import Pipeline
from agent.models import Conversation
from view.config import SOLUTIONS, PROBLEM, INSIGHTS, initialize_session_state
from view.ui_components import (
    load_css_styles,
    render_welcome_page,
    render_header,
    render_sidebar,
    render_right_sidebar,
    render_main_content,
)
from view.audio_processing import (
    setup_audio_capture,
    process_audio_frames,
    TranscriptionThread,
)


def main():
    """Main application function."""
    # Configure page
    st.set_page_config(layout="wide")
    st.markdown(load_css_styles(), unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()
    transcription_thread = None

    # Initialize conversation if not already done
    if st.session_state.conversation is None:
        st.session_state.conversation = Conversation(
            problem_text=PROBLEM,
            solutions=SOLUTIONS,
            insights=INSIGHTS,
            background_info={},
            chat_history=[],
        )
        st.session_state.pipeline = Pipeline()

    # Initial welcome page
    if not st.session_state.started:
        if render_welcome_page():
            st.session_state.started = True
            st.rerun()
    else:
        # Main application layout
        render_header()

        # Sidebar with solutions and transcript
        with st.sidebar:
            render_sidebar(st.session_state.conversation)

        # Main content columns
        main_content, right_sidebar = st.columns([4, 1], gap="small")

        with right_sidebar:
            if render_right_sidebar(st.session_state.conversation):
                # Finish button was clicked
                st.session_state.started = False
                st.session_state.conversation = None
                st.session_state.pipeline = None
                st.session_state.transcription_text = ""
                st.rerun()

        with main_content:
            render_main_content(st.session_state.conversation)

        # Add space and divider
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.write("**Voice Recording Status:** Active")

        # WebRTC streaming container
        webrtc_container = st.container()
        webrtc_ctx, current_time = setup_audio_capture(webrtc_container)

        # Audio processing loop
        while True:
            # Process audio and update transcription
            transcription_thread, current_time, should_continue = process_audio_frames(
                webrtc_ctx, current_time
            )

            # Store transcription thread in session state to access it on next run
            if transcription_thread:
                st.session_state.transcription_thread = transcription_thread

            # If we should stop and rerun the app
            if not should_continue:
                st.rerun()


if __name__ == "__main__":
    main()
