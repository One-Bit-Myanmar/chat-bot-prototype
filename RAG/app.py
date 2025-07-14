from fastapi import FastAPI
from pydantic import BaseModel
import RAG

app = FastAPI()

API_KEY = "Your key"
rag = RAG.LangChainRAG(api_key=API_KEY)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(req: QuestionRequest):
    try:
        answer = rag.ask(req.question)
        return {"question": req.question, "answer": answer}
    except Exception as e:
        return {"error": str(e)}
