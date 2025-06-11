from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_form import generer_story
from rag_chat import chat

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Form API models
class FormRequest(BaseModel):
    product: str
    audience: str
    tone: str
    

@app.post("/api/form/ask")
async def form_ask(req: FormRequest):
    story = generer_story(req.product, req.audience, req.tone)
    return {"story": story}

# Chat API models
class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat/message")
async def chat_message(req: ChatRequest):
    response = chat(req.message)
    return {"response": response}
