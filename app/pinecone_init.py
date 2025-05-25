import pinecone
from app.config import PINECONE_API_KEY, PINECONE_ENV, INDEX_NAME

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=1536)
index = pinecone.Index(INDEX_NAME)
