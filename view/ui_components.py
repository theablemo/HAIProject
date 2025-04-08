import streamlit as st
import streamlit.components.v1 as components


def load_css_styles():
    """Return the CSS styles for the application."""
    return """
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
        div.stButton > button {
            background-color: #c077fc;
            color: white;
            padding: 14px 20px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
        }
        div.stButton > button:hover {
            background-color: #7c04de;
        }
        /* Sound wave animation */
        .sound-wave {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 60px;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .wave-bar {
            width: 4px;
            height: 20px;
            background: #7c04de;
            margin: 0 2px;
            animation: wave 1s ease-in-out infinite;
        }
        .wave-bar:nth-child(2) { animation-delay: 0.1s; }
        .wave-bar:nth-child(3) { animation-delay: 0.2s; }
        .wave-bar:nth-child(4) { animation-delay: 0.3s; }
        .wave-bar:nth-child(5) { animation-delay: 0.4s; }
        .wave-bar:nth-child(6) { animation-delay: 0.5s; }
        .wave-bar:nth-child(7) { animation-delay: 0.6s; }
        .wave-bar:nth-child(8) { animation-delay: 0.7s; }
        .wave-bar:nth-child(9) { animation-delay: 0.8s; }
        .wave-bar:nth-child(10) { animation-delay: 0.9s; }
        @keyframes wave {
            0%, 100% { height: 20px; }
            50% { height: 50px; }
        }
        /* Countdown timer */
        .countdown-container {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #7c04de;
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        </style>
    """


def get_timer_html():
    """Return the HTML/JS for the timer component."""
    return """
    <div id="timer" style="color: white; font-size: 24px;">
        <div>Time left for new updates:</div>
        <div><span id="counter">20</span> seconds</div>
    </div>
    <script>
        let count = 20;
        const counterElement = document.getElementById('counter');

        function updateTimer() {
            counterElement.innerText = count;
            count--;
            if (count < 0) {
                count = 20;
            }
        }

        setInterval(updateTimer, 1000);
    </script>
    """


def render_sound_wave():
    """Render the animated sound wave component."""
    st.markdown(
        """
        <div class="sound-wave" style="background-color: transparent;">
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_vertical_divider():
    """Render a vertical divider used in the insights section."""
    return st.html(
        """
        <div class="divider-vertical-line"></div>
        <style>
            .divider-vertical-line {
                border-left: 2px solid rgba(49, 51, 63, 0.2);
                height: 320px;
                margin: auto;
            }
        </style>
        """
    )


def render_welcome_page():
    """Render the welcome page with start button."""
    st.markdown("<h1 style='text-align: center;'>Welcome</h1>", unsafe_allow_html=True)
    _, col2, __ = st.columns([1, 2, 1])
    with col2:
        return st.button("Start", key="start")


def render_timer():
    """Render the countdown timer."""
    components.html(get_timer_html(), height=70)


def render_header():
    """Render the header with sound wave and timer."""
    wave_col, countdown_col = st.columns([2, 1])
    with wave_col:
        render_sound_wave()
    with countdown_col:
        render_timer()
    st.markdown("---")  # Add divider


def render_sidebar(conversation):
    """Render the sidebar with solutions and transcript."""
    st.markdown(
        "<h2 style='text-align: center;'>Solutions/Decisions</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    for solution in conversation.solutions:
        st.markdown(f"**{solution.title}**  \n{solution.subtitle}")
    with st.expander("📃 Transcript", expanded=True):
        st.write(st.session_state.transcription_text)


def render_right_sidebar(conversation):
    """Render the right sidebar with finish button and information."""
    # Finish Session button
    finish_clicked = st.button("Finish Session")
    st.markdown("---")
    st.markdown(
        "<h3 style='color: #7c04de;'><ins>Information</ins></h3>",
        unsafe_allow_html=True,
    )
    for key, value in conversation.background_info.items():
        st.markdown(f"**{key}:** {value}")
    return finish_clicked


def render_main_content(conversation):
    """Render the main content area with problem and insights."""
    # Heading
    st.markdown(
        f"<h2 style='color: #7c04de;'><ins>Problem Discussion</ins></h2> \n<h3>{conversation.problem_text}</h3>",
        unsafe_allow_html=True,
    )

    # Display insights in tabs
    if conversation.insights:
        tabs = st.tabs([f"Insight {i+1}" for i in range(len(conversation.insights))])
        for tab, insight in zip(tabs, conversation.insights):
            with tab:
                insight_column, divider, source_column = st.columns([3, 0.1, 1])
                with insight_column:
                    st.markdown(
                        f"<p style='font-size: 2em;'>{insight.text}</p>",
                        unsafe_allow_html=True,
                    )
                with divider:
                    render_vertical_divider()
                with source_column:
                    st.markdown(
                        "<h3 style='font-size: 2em; color: #7c04de;'><ins>Sources</ins></h3>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"{', '.join(insight.sources)}")
                if insight.vega_lite_spec:
                    st.vega_lite_chart(insight.vega_lite_spec, use_container_width=True)
