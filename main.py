from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chat import chat
import uvicorn

app = FastAPI()

class ChatRequest(BaseModel):
    query: str
    session_id:str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        print(request.query,request.session_id)
        response = chat(request.query,request.session_id)
        print(response.content)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
