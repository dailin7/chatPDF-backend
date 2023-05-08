from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv

def load_env():
    load_dotenv()
    OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
    return {"OPEN_AI_KEY": OPEN_AI_KEY}

#TODO
# take a deeper look at this part
# how to process a pdf file(embedding model)

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


env = load_env()
qa = load_chain()
print(qa.run('what is babyAGI'))



