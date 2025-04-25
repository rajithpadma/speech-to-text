import os
import torch
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import streamlit as st

# Use current working directory to save/load model
SCRIPT_DIR = os.getcwd()  # Get current working directory
MODEL_PATH = os.path.join(SCRIPT_DIR, "wav2vec2_model")  # Path to save the model

# Create the directory if it doesn't exist
os.makedirs(MODEL_PATH, exist_ok=True)

# Load or download model to current directory
def load_or_download_model():
    # Download and save model locally if not already saved
    if not os.path.exists(os.path.join(MODEL_PATH, "pytorch_model.bin")):
        st.write("🌐 Downloading model from Hugging Face...")
        
        # Load the model and processor from Hugging Face
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        
        # Save the model and processor locally as .bin and other necessary files
        processor.save_pretrained(MODEL_PATH)  # Save processor files
        model.save_pretrained(MODEL_PATH)  # Save model weights (pytorch_model.bin) and config
        
        st.write(f"💾 Model saved locally to: {MODEL_PATH}")
    else:
        st.write(f"📦 Loading model from local directory: {MODEL_PATH}")
        processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)

    return processor, model

# Record audio from mic
def record_audio(duration=5, filename="mic_audio.wav"):
    st.write("🎤 Recording...")
    fs = 16000  # sample rate
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # wait until recording is finished
    write(filename, fs, audio)
    st.write(f"✅ Saved: {filename}")
    return filename

# Transcribe using Wav2Vec2
def transcribe_wav2vec2(file_path):
    processor, model = load_or_download_model()

    audio, _ = librosa.load(file_path, sr=16000)
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

# Streamlit UI
st.title("Speech to Text using Wav2Vec2")
st.write("This app records audio from your microphone and transcribes it into text using the Wav2Vec2 model.")

# Button to record audio
if st.button('Record Audio'):
    filename = record_audio(duration=5)  # record for 5 seconds
    transcription = transcribe_wav2vec2(filename)
    st.write(f"📝 Transcription: {transcription}")

