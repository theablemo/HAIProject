import queue

import pydub

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings


def main():
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        client_settings=ClientSettings(
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "audio": True,
            },
        ),
    )

    if "audio_buffer" not in st.session_state:
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

    status_indicator = st.empty()

    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            sound_chunk = pydub.AudioSegment.empty()
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
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break

    audio_buffer = st.session_state["audio_buffer"]

    if not webrtc_ctx.state.playing and len(audio_buffer) > 0:
        st.info("Writing wav to disk")
        audio_buffer.export("temp.wav", format="wav")

        # Reset
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()


if __name__ == "__main__":
    main()