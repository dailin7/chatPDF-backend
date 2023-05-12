from multiprocessing import Pool, Manager
from langchain.document_loaders import UnstructuredPDFLoader
import glob, os, qdrant_client, time, random, multiprocessing

class pdf_loader:
    def __init__(self):
        self.files = self.load_all_pdfs()
        
    def load_pdf(i, files, lock, pages):
        try:
            loader = UnstructuredPDFLoader(files[i])
            lock.acquire()
            page=loader.load_and_split()
            pages.extend(page)
            lock.release()
        except:
            print("PDF files loading failed")

    def load_all_pdfs(self):
        if __name__=='main':
            pages = Manager().list()
            files = Manager().dict()
            lock = Manager().Lock()
            index = 0

            for file in glob.glob("test_data/*.pdf"):
                files[index] = file
                index+=1
            p = Pool(len(files))

            for i in range(len(files)):
                p.apply_async(self.load_pdf, args=(i, files, lock, pages))
            p.close()
            p.join()
            return list(pages)




     
