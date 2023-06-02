from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os, qdrant_client
from dotenv import load_dotenv
from .sources import Sources

class Chain:
    env: dict
    qa: ConversationalRetrievalChain
    sources: Sources

    def load_env(self):
        load_dotenv()
        OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
        self.env = {"OPEN_AI_KEY": OPEN_AI_KEY}
    
    def load_chain(self):
        self.load_env()
        embeddings = OpenAIEmbeddings(openai_api_key=self.env["OPEN_AI_KEY"])
        if(not os.path.exists("backend/data/local_qdrant/collection/my_documents")):
            #load and split input files        
            self.sources.load_all_pdf_3()
            
            #embed the input files and load it to DB
            qdrant = Qdrant.from_documents(
                self.sources.texts, 
                embeddings, 
                path="backend/data/local_qdrant",
                collection_name="my_documents",
            )
        else:
            client = qdrant_client.QdrantClient(
                path="backend/data/local_qdrant",
            )
            qdrant = Qdrant(
                client=client, 
                collection_name="my_documents", 
                embeddings=embeddings
            )

        #construct a qa chain with customized llm and db
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        #chain = load_qa_chain(OpenAI(model_name="text-ada-001", openai_api_key=env["OPEN_AI_KEY"], temperature=0), chain_type="map_reduce")
        self.qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=self.env["OPEN_AI_KEY"], temperature=0), qdrant.as_retriever(), memory=memory, return_source_documents=True)
    
