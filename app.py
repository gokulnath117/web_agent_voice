import streamlit as st
import sounddevice as sd
import scipy.io.wavfile
import requests
import io

st.set_page_config(
    page_title="Stock Market Voice Assistant",
    page_icon="🎤",
    layout="centered"
)

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎤 Stock Market Voice Assistant")
st.markdown("""
    Ask questions about stocks using your voice! You can:
    - Get stock prices and historical data
    - Get latest stock market news
    - Ask general stock market questions
""")

st.subheader("🎙️ Voice Input")
duration = st.slider("Recording Duration (seconds)", 1, 10, 5)

if st.button("Start Recording", key="record_button"):
    st.info("🎙️ Recording... Speak now!")
    fs = 16000
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    wav_io = io.BytesIO()
    scipy.io.wavfile.write(wav_io, fs, audio)
    wav_io.seek(0)

    files = {"file": ("recording.wav", wav_io, "audio/wav")}
    response = requests.post("http://127.0.0.1:8000/transcribe/", files=files)

    if response.status_code == 200:
        data = response.json()

        if "error" in data:
            st.error(data["error"])
        else:
            st.markdown