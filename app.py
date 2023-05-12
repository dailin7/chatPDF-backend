from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from multiprocessing import Pool, Manager
import glob, os, qdrant_client, time, random, multiprocessing
import pdf_loader 
from dotenv import load_dotenv

'''
load env variables from .env file
'''
def load_env():
    load_dotenv()
    OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
    return {"OPEN_AI_KEY": OPEN_AI_KEY}

'''
generate prompts for llm
'''
#TODO: implement the method
def generate_prompt():
    return None

#TODO: fine-tune the parameters of chain, embedding model, llm, and db
def load_chain():
    embeddings = OpenAIEmbeddings(openai_api_key=env["OPEN_AI_KEY"])
    if(not os.path.exists("../chatPDF/data/local_qdrant/collection/my_documents")):
        #load and split input files        
        texts = pages
        
        #embed the input files and load it to DB
        qdrant = Qdrant.from_documents(
        texts, embeddings, 
        path="../chatPDF/data/local_qdrant",
        collection_name="my_documents",
         )
    else:
        client = qdrant_client.QdrantClient(
        path="../chatPDF/data/local_qdrant",
        )
        qdrant = Qdrant(
            client=client, collection_name="my_documents", 
            embeddings=embeddings
        )

    #construct a qa chain with customized llm and db
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    #chain = load_qa_chain(OpenAI(model_name="text-ada-001", openai_api_key=env["OPEN_AI_KEY"], temperature=0), chain_type="map_reduce")
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=env["OPEN_AI_KEY"], temperature=0), qdrant.as_retriever(), memory=memory)

    return qa

def ask_and_answer():
    qa = load_chain()

    print("Type exit to quite")
    print("-"*30)
    while (True):
        question = input("You: ")
        if (question == "exit"):
            break
        
        answer = qa.run(question)
        print("Chatbot: " + answer + "\n")
    print("program terminated by user")

def load_pdf(i, files, lock, pages):
    try:
        loader = UnstructuredPDFLoader(files[i])
        lock.acquire()
        page=loader.load_and_split()
        pages.extend(page)
        lock.release()
    except Exception as e:
        print(f"PDF file {files[i]} loading failed due to error: {e}")

'''
load local pdfs and split them
'''
#TODO: fine-tune the parameters of loader and splitter
def load_all_pdfs():
    pages = Manager().list()
    files = Manager().dict()
    lock = Manager().Lock()
    index = 0

    for file in glob.glob("test_data/*.pdf"):
        files[index] = file
        index+=1

    p = Pool(len(files))

    for i in range(len(files)):
        p.apply_async(load_pdf, args=(i, files, lock, pages))

    p.close()
    p.join()

    return(list(pages))

if __name__=='__main__':
    env = load_env()
    pages = load_all_pdfs()
    ask_and_answer()




