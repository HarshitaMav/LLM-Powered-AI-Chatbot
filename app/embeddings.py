from langchain.embeddings import OpenAIEmbeddings
from app.config import OPENAI_API_KEY

def get_embeddings():
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
