from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from app.embeddings import get_embeddings
from app.pinecone_init import index
import os

def create_retriever():
    loader = TextLoader("data/support_faq.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    vectordb = Pinecone.from_documents(docs, embeddings, index_name="support-chatbot")
    retriever = vectordb.as_retriever()
    return retriever

def get_qa_chain():
    retriever = create_retriever()
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
