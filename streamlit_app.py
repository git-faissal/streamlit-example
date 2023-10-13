import streamlit as st 
from PyPDF2 import PdfReader
import requests
#importation des bibliotheque de synthese de presse en ligne
import pandas as pd
import requests
import datetime
import json
#IMPORTATION BIBLIOTHEQUE LANGCHAIN
#Import des bibliotheque de langchain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#OPENAI_API_KEY=""
import os
os.environ['OPENAI_API_KEY'] = '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'
#import des bibliotheques langchain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
import textwrap
from langchain.chains import ConversationChain
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
import unicodedata
#Fin IMPORTATION
#import des package

st.set_page_config(layout="wide")
@st.cache_resource
#Fonction de resume de texte avec txtai
def summary_text(text):
    summary = Summary()
    text = (text)
    result = summary(text)
    return result

def correct_spelling(text):
    tool = LanguageTool('fr')
    corrected_text = tool.correct(text)
    return corrected_text

#Fonction Resume Langchain
def summarizeLangChain(input_text):
    #Definition de ma cle API
    os.environ['OPENAI_API_KEY'] = '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'
    llm = OpenAI(temperature=0)
    """Generates a summary of the given text using LangChain.

    Args:
        text: The text to be summarized.

    Returns:
        A string containing the summary of the given text.
    """

    # Convert the text to a LangChain Document object
    doc = Document(text=text, page_content=text)

    # Define prompt
    prompt_template = """Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    # Define LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    # Exécutez la chaîne de traitement
    summary = llm_chain.run(text)
    #traduction francais
    result = traduction_francais(summary)
    #CORRECTION DU RESUME
    correctedText = correct_spelling(result)
    return correctedText

#Fonction extraction du texte d'un document pdf
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return text

#Fonction appel aux modele de resume de texte en utilisant l'appel de l'API
#[0]['generated_text']
#def query(payload):
    #API_URL = "https://api-inference.huggingface.co/models/tuner007/pegasus_summarizer"
    #API_URL = "https://api-inference.huggingface.co/models/gpt2"
    #headers = {"Authorization": "Bearer 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"}
    #response = requests.post(API_URL, headers=headers, json=payload)
    #return response.json()

# Fonction pour obtenir le résumé d'un texte
#def get_summary(text):
    #resume = query({
    #    "inputs": text
    #})
    #return resume[0]['summary_text']

#Fonction appel aux modele de resume de texte en utilisant l'appel de l'API
def query(payload):
    API_TOKEN="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"
    API_URL ="https://api-inference.huggingface.co/models/tuner007/pegasus_summarizer"
    #API_URL = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

#fonction appel de resume
def get_summary(text):
    data = query(text)
    if data and isinstance(data, list) and data[0].get('summary_text') is not None:
        return data[0]['summary_text']
    else:
        error = "Erreur survenue lors de l'appel de l'API. Veuillez recommencer svp !!!"
        return error

        
#Fonction de resume de texte avec Pegasus Turned
def get_response(input_text):
    model_name = 'tuner007/pegasus_summarizer'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    max_input_length = 1024  # Maximum length of the input text
    desired_summary_length = len(input_text) // 2  # Calculate desired summary length
    batch = tokenizer([input_text], truncation=True, padding='longest', max_length=max_input_length, return_tensors="pt").to(torch_device)
    gen_out = model.generate(
        **batch,
        max_length=desired_summary_length,  # Set the max length for the summary
        num_beams=5,
        num_return_sequences=1,
        temperature=1.5
    )
    output_text = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    return output_text


choice = st.sidebar.selectbox(
    "Choisir",
    [
       
        "Resumer un Texte",
        "Resumer un document"
    ]
)

def run_app():
    if choice == "Resumer un Texte":
        st.subheader("NB: Dans cette zone, vous pourrez tester le modèle pour résumer vos reportages effectués sur le terrain. Ainsi, vous pourrez permettre à vos lecteurs de lire vos résumés et de se renseigner plus rapidement.")
        st.subheader("Resume de texte avec LangChain")
        input_text = st.text_area("Entrez votre texte ici")
        if st.button("Texte Resume"):
             col1, col2 = st.columns([1,1])
             with col1:
                st.markdown("***Votre texte entrez***")
                st.info(input_text)
             with col2:
                #result = get_response(input_text)
                result = summarizeLangChain(input_text)
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
                    summary_result = get_summary(result)
                    st.success(summary_result)
            

if __name__ == '__main__':
    run_app()
