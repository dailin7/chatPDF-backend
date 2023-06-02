from langchain.document_loaders import UnstructuredPDFLoader
import glob

class Sources:
    texts: list

    def load_all_pdf_3(self):
        for file in glob.glob("backend/test_data/*.pdf"):
            loader = UnstructuredPDFLoader(file)
            self.texts+=loader.load_and_split()

#TODO
    def add_pdf(self):
        pass