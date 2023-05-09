from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

import glob, os
from dotenv import load_dotenv

def load_env():
    load_dotenv()
    OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
    return {"OPEN_AI_KEY": OPEN_AI_KEY}

#TODO: fine-tune the parameters of loader and splitter
def load_pdf():
    pages = []
    for file in glob.glob("test_data/*.pdf"):
        loader = UnstructuredPDFLoader(file)
        pages+=loader.load_and_split()
    return pages
    
#TODO: implement the method
def generate_prompt():
    return None

#TODO: fine-tune the parameters of chain, embedding model, llm, and db
def load_chain():
    texts = load_pdf()

    #embed the input files and load it to DB
    embeddings = OpenAIEmbeddings(openai_api_key=env["OPEN_AI_KEY"])
    db = DeepLake.from_documents(texts, embeddings)

    #construct a qa chain with customized llm and db
    chain = load_qa_chain(OpenAI(openai_api_key=env["OPEN_AI_KEY"], temperature=0), chain_type="map_reduce")
    qa = RetrievalQA(combine_documents_chain=chain, retriever=db.as_retriever())

    return qa

def ask_and_answer():
    print("Type exit to quite")
    print("-"*30)
    while (True):
        question = input("You: ")
        if (question == "exit"):
            break
        
        answer = qa.run(question)
        print("Chatbot: " + answer + "\n")
    print("program terminated by user")

env = load_env()
qa = load_chain()
ask_and_answer()




