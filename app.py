from typing import Any, Dict, List
from io import BytesIO
import re
import logging

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
import streamlit as st
from streamlit_chat import message
import glob, os
import qdrant_client
from dotenv import load_dotenv
from pypdf import PdfReader


def load_env():
    load_dotenv()
    OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
    return {"OPEN_AI_KEY": OPEN_AI_KEY}


def handle_file(file):
    filename = file.filename
    logging.info("[handle_file] Handling file: {}".format(filename))

    try:
        extracted_text = extract_text_from_file(file)
    except ValueError as e:
        logging.error("[handle_file] Error extracting text from file: {}".format(e))
        raise e


def extract_text_from_file(file):
    """Return the text content of a file."""
    pass


# TODO: fine-tune the parameters of loader and splitter
def load_pdf():
    pages = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        separators=[" ", ",", "\n"],
    )
    for file in glob.glob("test_data/*.pdf"):
        # loader = UnstructuredPDFLoader(file)
        # pages += loader.load_and_split(text_splitter=text_splitter)
        # get file name for file
        file_name = file.split("/")[-1].split(".")[0]
        pdf = parse_pdf(file)
        loader = text_to_docs(pdf, file_name)
        pages += loader
    return pages


# @st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output


# @st.cache_data
def text_to_docs(text: str, file_name: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata[
                "source"
            ] = f"{file_name}-{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


# TODO: implement the method
def generate_prompt():
    return None


# TODO: fine-tune the parameters of chain, embedding model, llm, and db
# @st.cache_resource
def load_chain():
    embeddings = OpenAIEmbeddings(openai_api_key=env["OPEN_AI_KEY"])
    if not os.path.exists("../chatPDF/data/local_qdrant/collection/my_documents"):
        # load and split input files
        texts = load_pdf()

        # embed the input files and load it to DB
        qdrant = Qdrant.from_documents(
            texts,
            embeddings,
            path="../chatPDF/data/local_qdrant",
            collection_name="my_documents",
        )
    else:
        client = qdrant_client.QdrantClient(
            path="../chatPDF/data/local_qdrant",
        )
        qdrant = Qdrant(
            client=client, collection_name="my_documents", embeddings=embeddings
        )

    # construct a qa chain with customized llm and db
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    # chain = load_qa_chain(OpenAI(model_name="text-ada-001", openai_api_key=env["OPEN_AI_KEY"], temperature=0), chain_type="map_reduce")
    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(openai_api_key=env["OPEN_AI_KEY"], temperature=0),
        qdrant.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
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
    print("-" * 30)
    while True:
        question = input("You: ")
        if question == "exit":
            break

        answer = chain({"question": question})
        print("Chatbot: " + answer["answer"] + "\n")
        print(
            "------------------------------SOURCE-----------------------------" + "\n"
        )
        print(answer["source_documents"])

    print("program terminated by user")


env = load_env()
chain = load_chain()
ask_and_answer()


# # st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
# st.header("LangChain Demo")

# if "generated" not in st.session_state:
#     st.session_state["generated"] = []

# if "past" not in st.session_state:
#     st.session_state["past"] = []


# def get_text():
#     input_text = st.text_input("You: ", "Hello, how are you?", key="input")
#     return input_text


# user_input = get_text()

# if user_input:
#     output = chain({"question": user_input})
#     st.session_state.past.append(user_input)
#     # output['answer'] + '\n' +
#     st.session_state.generated.append(
#         output["answer"]
#         + "\n"
#         + "----------------------------- SOURCRE -----------------------"
#         + "\n"
#         + " ".join(set([x.metadata["source"] for x in output["source_documents"]]))
#     )
#     # output['source_documents']
# if st.session_state["generated"]:
#     for i in range(len(st.session_state["generated"]) - 1, -1, -1):
#         message(st.session_state["generated"][i], key=str(i))
#         message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
