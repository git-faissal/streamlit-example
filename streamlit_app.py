import streamlit as st
from PyPDF2 import PdfReader
import requests
import pandas as pd
import datetime
import json
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from vosk import Model, KaldiRecognizer
import soundfile as sf

from pydub import AudioSegment
import io

import os
import ffmpeg
import numpy as np
import os
import ffmpeg


import streamlit as st
import assemblyai as aai






st.set_page_config(layout="wide")

# Fonction de résumé de texte avec txtai
@st.cache_resource
def summary_text(text):
    summary = Summary()
    result = summary(text)
    return result


# Configuration de l'API Key
aai.settings.api_key = "42b6f7e917114668b99a01927dc49d8c"

# Fonction pour transcrire un fichier audio en texte avec AssemblyAI
def transcribe_audio_assemblyai(audio_file):
    try:
        # Transcription de l'audio
        transcript = aai.Transcriber().transcribe(audio_file)
        return transcript.text
    except Exception as e:
        return f"Erreur lors de la transcription : {str(e)}"


# Fonction pour extraire du texte d'un document PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return text

# Fonction d'appel à l'API pour la génération de résumé
def query(payload):
    API_TOKEN = "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"
    API_URL = "https://api-inference.huggingface.co/models/tuner007/pegasus_summarizer"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Fonction pour obtenir le résumé d'un texte via l'API
def get_summary(text):
    data = query({"inputs": text})
    if data and isinstance(data, list) and 'summary_text' in data[0]:
        return data[0]['summary_text']
    else:
        return "Erreur survenue lors de l'appel de l'API. Veuillez réessayer."

# Fonction pour obtenir un résumé avec Pegasus directement
def get_response(input_text):
    model_name = 'tuner007/pegasus_summarizer'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    
    batch = tokenizer([input_text], truncation=True, padding='longest', max_length=1024, return_tensors="pt").to(torch_device)
    gen_out = model.generate(**batch, max_length=150, num_beams=5, num_return_sequences=1)
    
    output_text = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    return output_text[0]

# Menu de sélection pour différentes fonctionnalités
choice = st.sidebar.selectbox(
    "Choisir",
    [
        "Résumé d'un texte",
        "Transcription Audio-Text",
        "Résumé d'un document"
    ]
)

# Application principale
def run_app():
    if choice == "Résumé d'un texte":
        st.subheader("Résumé de texte avec Pegasus_Summarizer")
        input_text = st.text_area("Entrez votre texte ici")
        if st.button("Résumer le texte"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("***Texte saisi***")
                st.info(input_text)
            with col2:
                result = get_summary(input_text)
                st.markdown("***Résumé généré***")
                st.success(result)

    elif choice == "Résumé d'un document":
        st.subheader("Résumé d'un document PDF")
        input_file = st.file_uploader("Charger un document PDF", type=['pdf'])
        if input_file is not None:
            if st.button("Résumer le document"):
                with open("doc_file.pdf", "wb") as f:
                    f.write(input_file.getbuffer())
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Texte extrait du document**")
                    extracted_text = extract_text_from_pdf("doc_file.pdf")
                    st.info(extracted_text)
                with col2:
                    summary_result = get_summary(extracted_text)
                    st.markdown("**Résumé du document**")
                    st.success(summary_result)

    elif choice == "Transcription Audio-Text":
        st.subheader("Transcription Audio en Texte")
        input_file = st.file_uploader("Chargez un fichier audio (.mp4)", type=['mp4'])
        if input_file is not None:
            if st.button("Transcrire audio"):
                with open("audio_file.mp4", "wb") as f:
                    f.write(input_file.getbuffer())
                
                result = transcribe_audio_assemblyai("audio_file.mp4")
                st.markdown("**Texte transcrit**")
                st.success(result)

if __name__ == '__main__':
    run_app()


