from multiprocessing import Pool, Manager
from langchain.document_loaders import UnstructuredPDFLoader
import glob, os, qdrant_client, time, random, multiprocessing
import concurrent.futures


#1: multiprocess(multiprocessing.Pool)
#2: multiprocess(concurrent.futures.Executor)
#3: linear process
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

def load_pdf_2(file):
    try:
        loader = UnstructuredPDFLoader(file)
        page = loader.load_and_split()
        return page
    except Exception as e:
        print(f"PDF file {file} loading failed due to error: {e}")

def load_all_pdf_2():
    files = list()
    for file in glob.glob("test_data/*.pdf"):
            files.append(file)

    with concurrent.futures.ProcessPoolExecutor(int(os.cpu_count()/2)) as executor:
        pages = list(executor.map(load_pdf_2, files))

    # Flatten the list of lists into a single list
    pages = [string for sublist in pages for string in sublist]
    return pages

def load_all_pdf_3():
    pages = []
    for file in glob.glob("test_data/*.pdf"):
        loader = UnstructuredPDFLoader(file)
        pages+=loader.load_and_split()
    return pages

'''
if __name__=='__main__':
    multiprocess...
'''

def test(num):
    try:
        result = [num, num*2, num*3]
        return result
    except Exception as e:
        print("failed due to error: {e}")

if __name__=='__main__':
    nums = [1, 2, 3, 4, 5, 6, 7]

    time_1 = time.time()
    p = Pool(int(os.cpu_count()/2)) #faster, tested
    result_1 = p.map(test, nums)
    print(str(time.time() - time_1))

    time_2 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers = int(os.cpu_count()/2)) as executor:
        result_2 = list(executor.map(test, nums))
    print(str(time.time() - time_2))

    print(result_1)
    print(result_2)








     
