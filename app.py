""" Run app and get responses """
import streamlit as st
from qa_chain import get_answer

st.title('University FAQ AI Assistant')

question = st.text_input('Ask a question about university services:')
if question:
    answer = get_answer(question)
    st.write(answer)
