import streamlit as st
from streamlit.components.v1 import html
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

def init_page():
    st.set_page_config(page_title='chatPDF')
    st.title('chatPDF')
    with st.sidebar:
        uploaded_files = st.file_uploader('Choose your PDF files', type=['pdf', 'PDF'], accept_multiple_files=True)

@st.cache_resource
def load_chain():
    llm = OpenAI(temperature=0, openai_api_key='')
    chain = ConversationChain(llm=llm)
    return chain

def init_conversation():
    if 'hist_questions' not in st.session_state:
        st.session_state['hist_questions'] = []

    if 'hist_responses' not in st.session_state:
        st.session_state['hist_responses'] = []

init_page()
chain = load_chain()
init_conversation()

def ask_and_answer():
    question = st.text_input(label = 'Type in your question', value='')

    if question:
        output = chain.run(input=question)
        st.session_state['hist_questions'].append(question)
        st.session_state['hist_responses'].append(output)

def show_conversation():
    if st.session_state['hist_responses']:
        for i in range(len(st.session_state['hist_responses']) - 1):
            message(st.session_state['hist_responses'][i])
            message(st.session_state['hist_questions'][i], is_user=True)

ask_and_answer()
show_conversation()



