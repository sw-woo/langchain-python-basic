import streamlit as st
from langchain_ollama.llms import OllamaLLM
import pandas as pd
from pandasai import SmartDataframe

llm = OllamaLLM(model="llama3.1:8b")

st.title("Data analysis with PandasAI")

uploader_file = st.file_uploader("upload a your csv file", type=["csv"])

if uploader_file is not None:
    data = pd.read_csv(uploader_file)
    st.write(data.head(5))
    df = SmartDataframe(data, config={"llm": llm})
    prompt = st.text_area("Enter your prompt")

    if st.button('Analyze'):
        if prompt:
            with st.spinner('Analyzing...'):
                st.write(df.chat(prompt))
        else:
            st.warning('Please enter a prompt')
