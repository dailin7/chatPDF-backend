import streamlit as st
from streamlit.components.v1 import html
from streamlit_chat import message
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv

def init_page():
    st.set_page_config(page_title='chatPDF')
    st.title('chatPDF')
    with st.sidebar:
        uploaded_files = st.file_uploader('Choose your PDF files', type=['pdf', 'PDF'], accept_multiple_files=True)
init_page()

@st.cache_data
def load_env():
    load_dotenv()
    OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
    return {"OPEN_AI_KEY": OPEN_AI_KEY}

#TODO
# take a deeper look at this part
# how to process a pdf file(embedding model)
@st.cache_resource
def load_chain():
    #load and split input files
    loader = TextLoader("../chatPDF/test_date/sample.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    #embed the input files and load it to DB
    embeddings = OpenAIEmbeddings(openai_api_key=env["OPEN_AI_KEY"])
    db = DeepLake.from_documents(texts, embeddings)

    #construct a qa chain with customized llm and db
    chain = load_qa_chain(OpenAI(openai_api_key=env["OPEN_AI_KEY"], temperature=0), chain_type="map_reduce")
    qa = RetrievalQA(combine_documents_chain=chain, retriever=db.as_retriever())

    return qa

def init_conversation():
    if 'hist_questions' not in st.session_state:
        st.session_state['hist_questions'] = []

    if 'hist_responses' not in st.session_state:
        st.session_state['hist_responses'] = []

#TODO should the answer also be added into the DB or the llm for better context and thus better performance in long term?
def ask_and_answer():
    question = st.text_input(label = 'Type in your question', value=None)

    if question:
        ans = qa.run(question)
        st.session_state['hist_questions'].append(question)
        st.session_state['hist_responses'].append(ans)

def show_conversation():
    if st.session_state['hist_responses']:
        for i in range(len(st.session_state['hist_responses']) - 1):
            message(st.session_state['hist_responses'][i])
            message(st.session_state['hist_questions'][i], is_user=True)

env = load_env()
init_conversation()
qa = load_chain()
ask_and_answer()
show_conversation()



