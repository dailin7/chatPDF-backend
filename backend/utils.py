from typing import Any, Dict, List, Tuple
from io import BytesIO
import os
import logging
import re


import docx2txt
import fitz
from hashlib import md5

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)

from .sources import qdrant_client


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


def load_pdf(file):
    pdf = parse_pdf(file)
    loader = text_to_docs(pdf, file.name, 3000)
    return loader


def load_docx(file):
    extracted_text = docx2txt.process(file)
    txt = parse_txt(extracted_text)
    loader = text_to_docs(txt, file.name, 1000)
    return loader


def load_txt(file):
    txt = parse_txt(file)
    loader = text_to_docs(txt, file.name, 1000)
    return loader


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


def parse_txt(extracted_text: BytesIO) -> List[str]:
    extracted_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", extracted_text)
    # Fix newlines in the middle of sentences
    extracted_text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", extracted_text)
    # Remove multiple newlines
    extracted_text = re.sub(r"\n\s*\n", "\n\n", extracted_text)

    return extracted_text


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


def upsert_documents_to_qdrant(
    path: str,
    collection_name: str = "my_documents",
):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    # client = QdrantClient(path=path)
    # client.upsert
    qdrant_client.recreate_collection(
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


def generate_embeddings():
    pass


def format_source(source_documents: List[Document]):
    res = ""
    for i, doc in enumerate(source_documents):
        res += f"source {i + 1} \n"
        res += f'   page: {doc.metadata["source"]} \n'
        res += f"   content: {doc.page_content} \n"
    return res
