from typing import Any, Dict, List, Tuple
from io import BytesIO
import re
import logging
import docx2txt
import fitz
from hashlib import md5

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
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
import streamlit as st
from streamlit_chat import message
import glob, os
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from pypdf import PdfReader


@st.cache_data
def load_files(files) -> List[Document]:
    pages = []
    for file in files:
        filetype = file.type
        if filetype == "application/pdf":
            pages += load_pdf(file)
        elif (
            filetype
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            pages += load_docx(file)
        elif filetype == "text/plain":
            pages += load_txt(file)
        else:
            raise ValueError(f"File type {filetype} not supported")

    return pages


@st.cache_data
def load_pdf(file):
    pdf = parse_pdf(file)
    loader = text_to_docs(pdf, file.name, 3000)
    return loader


@st.cache_data
def load_docx(file):
    extracted_text = docx2txt.process(file)
    txt = parse_txt(extracted_text)
    loader = text_to_docs(txt, file.name, 1000)
    return loader


@st.cache_data
def load_txt(file):
    txt = parse_txt(file)
    loader = text_to_docs(txt, file.name, 1000)
    return loader


@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    logging.info(type(file))
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    output = []
    for page in pdf:
        text = page.get_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output


@st.cache_data
def parse_txt(extracted_text: BytesIO) -> List[str]:
    extracted_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", extracted_text)
    # Fix newlines in the middle of sentences
    extracted_text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", extracted_text)
    # Remove multiple newlines
    extracted_text = re.sub(r"\n\s*\n", "\n\n", extracted_text)

    return extracted_text


@st.cache_data
def text_to_docs(text: str, file_name: str, chunk_size: int) -> List[Document]:
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
            chunk_size=chunk_size,
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


@st.cache_resource
def upsert_documents_to_qdrant(
    path: str,
    collection_name: str = "my_documents",
):
    embeddings = OpenAIEmbeddings(openai_api_key=api)
    # client = QdrantClient(path=path)
    # client.upsert
    qdrant = Qdrant.from_documents(
        pages,
        embeddings,
        path=path,
        collection_name=collection_name,
    )

    # client = QdrantClient(path=path)

    # qdrant = Qdrant(
    #     client=client, collection_name=collection_name, embeddings=embeddings
    # )

    # texts = [x.page_content for x in pages]
    # metadata = [x.metadata for x in pages]
    # print(texts[0])
    # print("\n")
    # print(metadata[0])
    # qdrant.add_texts(
    #     texts=[x.page_content for x in pages], metadatas=[x.metadata for x in pages]
    # )
    return qdrant


# TODO: fine-tune the parameters of chain, embedding model, llm, and db
@st.cache_resource
def load_chain():
    # construct a qa chain with customized llm and db
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    # chain = load_qa_chain(OpenAI(model_name="text-ada-001", openai_api_key=env["OPEN_AI_KEY"], temperature=0), chain_type="map_reduce")
    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(openai_api_key=api, temperature=0),
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


def format_source(source_documents: List[Document]):
    res = ""
    for i, doc in enumerate(source_documents):
        res += f"source {i + 1} \n"
        res += f'   page: {doc.metadata["source"]} \n'
        res += f"   content: {doc.page_content} \n"
    return res


# ask_and_answer()


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
st.title("🤖 Personalized Bot with Memory 🧠 ")
st.markdown(
    """ 
        ####  🗨️ Chat with your PDF files 📜 with `Conversational Buffer Memory`  
        ----
        """
)


# Set up the sidebar
st.sidebar.markdown(
    """
    ### Steps:
    1. Upload PDF File
    2. Enter Your Secret Key for Embeddings
    3. Perform Q&A

    **Note : File content and API key not stored in any form.**
    """
)

uploaded_file = st.file_uploader(
    "**Upload Your Files**", type=["pdf", "docx", "txt"], accept_multiple_files=True
)
path = "../chatPDF/data/local_qdrant"
collection_name = "my_documents"
if uploaded_file:
    pages = load_files(uploaded_file)
    api = st.text_input(
        "**Enter OpenAI API Key**",
        type="password",
        placeholder="sk-",
        help="https://platform.openai.com/account/api-keys",
    )
    if api:
        qdrant = upsert_documents_to_qdrant(
            path=path,
            collection_name=collection_name,
        )

        qa = load_chain()
        # qa = RetrievalQA.from_chain_type(
        #     llm=OpenAI(openai_api_key=api),
        #     chain_type="map_reduce",
        #     retriever=qdrant.as_retriever(),
        # )
        # Set up the conversational agent

        if "generated" not in st.session_state:
            st.session_state["generated"] = []

        if "past" not in st.session_state:
            st.session_state["past"] = []

        # Allow the user to enter a query and generate a response
        query = st.text_input(
            "**What's on your mind?**",
            placeholder="Ask me anything from {}".format(collection_name),
        )

        if query:
            # with st.spinner("Generating Answer to your Query : `{}` ".format(query)):
            res = qa({"question": query})
            print(res)
            st.session_state.past.append(query)
            st.session_state.generated.append(
                res["answer"]
                + "\n"
                + "----------------------------- SOURCRE -----------------------"
                + "\n"
                + format_source(res["source_documents"])
                # + res["source_documents"][0].metadata["source"]
            )
            # st.info(res, icon="🤖")
        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        # Allow the user to view the conversation history and other information stored in the agent's memory

    # output['source_documents']

with st.sidebar:
    st.markdown("Shanghai Artificial Intelligence Research Institute Co., Ltd.")
