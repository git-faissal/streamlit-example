import streamlit as st 

import streamlit as st
from txtai.pipeline import Summary
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader

#importation des bibliotheque de synthese de presse en ligne
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset, load_metric
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
from transformers import pipeline, set_seed

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

import datetime


st.set_page_config(layout="wide")
@st.cache_resource
#Fonction de resume de texte avec txtai
def summary_text(text):
    summary = Summary()
    text = (text)
    result = summary(text)
    return result

#Fonction extraction du texte d'un document pdf
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return text




choice = st.sidebar.selectbox(
    "Choisir",
    [
       
        "Resumer un Texte",
        "Resumer un document"
    ]
)

if choice == "Resumer un Texte":
    st.subheader("NB: Dans cette zone, vous pourrez tester le modèle pour résumer vos reportages effectués sur le terrain. Ainsi, vous pourrez permettre à vos lecteurs de lire vos résumés et de se renseigner plus rapidement.")
    st.subheader("Resume de texte avec PegasusTurned")
    input_text = st.text_area("Entrez votre texte ici")
    if st.button("Texte Resume"):
         col1, col2 = st.columns([1,1])
         with col1:
             st.markdown("***Votre texte entrez***")
             st.info(input_text)
         with col2:
             #result = get_response(input_text)
             result = summary_text(input_text)
             st.markdown("***Texte Resume***")
             st.success(result)
             
elif choice == "Resumer un document":
    st.subheader("Document resume avec Pegasus")
    input_file = st.file_uploader("Charger le document", type=['pdf'])
    if input_file is not None:
        if st.button("Resume un document"):
            with open("doc_file.pdf", "wb") as f:
                f.write(input_file.getbuffer())
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown("**Extraction du texte de votre document**")
                extracted_text = extract_text_from_pdf("doc_file.pdf")
                st.info(extracted_text)
            with col2:
                result = extract_text_from_pdf("doc_file.pdf")
                st.markdown("**Document resume**")
                summary_result = summary_text(result)
                st.success(summary_result)
            
