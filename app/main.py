from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.chatbot import get_qa_chain

app = FastAPI()
qa_chain = get_qa_chain()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    try:
        answer = qa_chain.run(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
