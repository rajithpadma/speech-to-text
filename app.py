import os
import torch
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

SCRIPT_DIR = os.getcwd()
MODEL_PATH = os.path.join(SCRIPT_DIR, "wav2vec2_tiny_quantized")
os.makedirs(MODEL_PATH, exist_ok=True)

# Load/download & quantize model
@st.cache_resource
def load_model():
    quantized_path = os.path.join(MODEL_PATH, "quantized_model.pt")

    if not os.path.exists(quantized_path):
        st.info("🔄 Downloading & quantizing model...")
        processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wav2vec2_tiny_random_robust")
        model = Wav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2_tiny_random_robust")
        model.cpu()
        model_quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        torch.save(model_quantized, quantized_path)  # Save entire model
        processor.save_pretrained(MODEL_PATH)
    else:
        processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
        model_quantized = torch.load(quantized_path)  # Load entire model
        model_quantized.eval()

    return processor, model_quantized

# Record audio using mic
def record_audio(duration=5, filename="recorded.wav"):
    fs = 16000
    st.info("🎙️ Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    st.success(f"✅ Recorded and saved: {filename}")
    return filename

# Transcription
def transcribe(file_path, processor, model):
    audio, _ = librosa.load(file_path, sr=16000)
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

# -------------------- UI --------------------
st.set_page_config(page_title="🗣️ Wav2Vec2 Transcriber", layout="centered")
st.title("🎧 Tiny Wav2Vec2 Transcriber")

processor, model = load_model()

option = st.radio("Choose input method:", ("🎤 Record Audio", "📁 Upload File"))

if option == "🎤 Record Audio":
    if st.button("⏺️ Record 5s"):
        file_path = record_audio()
        transcription = transcribe(file_path, processor, model)
        st.subheader("📝 Transcription:")
        st.success(transcription)

elif option == "📁 Upload File":
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file is not None:
        with open("uploaded.wav", "wb") as f:
            f.write(uploaded_file.read())
        transcription = transcribe("uploaded.wav", processor, model)
        st.subheader("📝 Transcription:")
        st.success(transcription)
