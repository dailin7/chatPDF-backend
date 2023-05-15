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
import concurrent.futures


'''
load env variables from .env file
'''
def load_env():
    load_dotenv()
    OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
    return {"OPEN_AI_KEY": OPEN_AI_KEY}

def load_pdf_1(file):
    try:
        loader = UnstructuredPDFLoader(file)
        page = loader.load_and_split()
        return page
    except Exception as e:
        print(f"PDF file {file} loading failed due to error: {e}")

def load_all_pdf_1():
    files = list()
    for file in glob.glob("test_data/*.pdf"):
            files.append(file)

    p = Pool(int(os.cpu_count()/2))

    pages = p.map(load_pdf_1, files)
    p.close()

    # Flatten the list of lists into a single list
    pages = [string for sublist in pages for string in sublist]
    return pages

'''
generate prompts for llm
'''
#TODO: implement the method
def generate_prompt():
    return None

#TODO: fine-tune the parameters of chain, embedding model, llm(question generator/prompt), and db
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
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    #chain = load_qa_chain(OpenAI(model_name="text-ada-001", openai_api_key=env["OPEN_AI_KEY"], temperature=0), chain_type="map_reduce")
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=env["OPEN_AI_KEY"], temperature=0), qdrant.as_retriever(), memory=memory, return_source_documents=True)

    return qa

def ask_and_answer():
    qa = load_chain()

    print("Type exit to quite")
    print("-"*30)
    while (True):
        question = input("You: ")
        if (question == "exit"):
            break
        
        # answer = qa.run(question)
        answer = qa({"question": question})
        # print(answer['answer'] + '\n')
        print(answer)

        # print("Chatbot: " + answer['answer'] + "\n")
    print("program terminated by user")

env = load_env()
if __name__=='__main__':
    start = time.time()
    pages = load_all_pdf_1()
    print(int(time.time() - start))
    ask_and_answer()






