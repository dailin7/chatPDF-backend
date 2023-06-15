from langchain.document_loaders import UnstructuredPDFLoader
import glob
from qdrant_client import QdrantClient
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from .config import OPEN_AI_KEY, VECTOR_DB_PATH


class Sources:
    texts: list

    def load_all_pdf_3(self):
        self.texts = list()
        for file in glob.glob("backend/test_data/*.pdf"):
            loader = UnstructuredPDFLoader(file)
            self.texts += loader.load_and_split()

    # TODO
    def add_pdf(self):
        pass


embedding = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)
try:
    qdrant_client = QdrantClient(url=VECTOR_DB_PATH)
except Exception as e:
    print("Cannot connect to vector db")
