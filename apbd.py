import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

def similarity(text_pemda, bagan_akun, mod = model):
    # encode sentences to get their embeddings
    embedding1 = mod.encode(text_pemda, convert_to_tensor=True)
    embedding2 = bagan_akun
    # compute similarity scores of two embeddings
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2).cpu().tolist()[0]
    max_score = max(cosine_scores)

    #Output passages & scores
    return cosine_scores.index(max_score)

def main():
    """App with Streamlit"""
    st.set_page_config(layout="wide")
    st.title("Standardisasi Akun APBD")
    
    bagan_df = pd.read_excel('Standar Akun Level 6.xlsx')
    akun = st.text_input(label="Uraian Akun")
    model = open("akun_enc.pkl", "rb")
    akun_enc = joblib.load(model)
    a = similarity(akun, akun_enc)
    out = bagan_df.iloc[a]['standarsubrinci']

    st.title("Klasifikasi Akun: "+out)

if __name__=='__main__':
    main()
