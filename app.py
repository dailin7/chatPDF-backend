from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
import streamlit as st
from streamlit_chat import message
import glob, os
import qdrant_client
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
@st.cache_resource
def load_chain():
    embeddings = OpenAIEmbeddings(openai_api_key=env["OPEN_AI_KEY"])
    if(not os.path.exists("../chatPDF/data/local_qdrant/collection/my_documents")):
        #load and split input files
        texts = load_pdf()

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
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')
    #chain = load_qa_chain(OpenAI(model_name="text-ada-001", openai_api_key=env["OPEN_AI_KEY"], temperature=0), chain_type="map_reduce")
    qa =ConversationalRetrievalChain.from_llm(ChatOpenAI(openai_api_key=env["OPEN_AI_KEY"], temperature=0), qdrant.as_retriever(), memory =  memory, return_source_documents=True)
    # llm = ChatOpenAI(openai_api_key=env["OPEN_AI_KEY"], temperature=0)
    # question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    # doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")
    # chain = ConversationalRetrievalChain(
    # retriever=qdrant.as_retriever(),
    # question_generator=question_generator,
    # combine_docs_chain=doc_chain,
    # memory=memory
    # )

    return qa

def ask_and_answer():
    print("Type exit to quite")
    print("-"*30)
    while (True):
        question = input("You: ")
        if (question == "exit"):
            break
        
        answer =  chain({"question": question})
        print("Chatbot: " + answer['answer'] + "\n")
    print("program terminated by user")

env = load_env()
chain= load_chain()
# ask_and_answer()


#st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chain({"question": user_input})
    st.session_state.past.append(user_input)
    #output['answer'] + '\n' +
    st.session_state.generated.append(output['answer'] + '\n'  + 
        '----------------------------- SOURCRE -----------------------'+ '\n'
        + " ".join(set([x.metadata['source'] for x in output['source_documents']])))
    #output['source_documents']
if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

