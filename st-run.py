import streamlit as st
from PIL import Image
import pandas as pd



# streamlit app
st.title("PDF to CSV Converter")

st.subheader("Search PDFs")

with st.form('searchForm'):
    pdf_files = st.file_uploader("Upload PDF Files:",accept_multiple_files=True)

    