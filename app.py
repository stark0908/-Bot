from fastapi import FastAPI, HTTPException
import requests
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn  # ✅ Required for running FastAPI properly

app = FastAPI()

# ✅ Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Hugging Face API Setup
HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# ✅ Define request format
class ChatRequest(BaseModel):
    user_input: str

@app.get("/")
def home():
    return {"message": "Dhruv Rathee Chatbot API is running!"}

# ✅ Chat endpoint with improved response formatting
@app.post("/chat")
async def chat(req: ChatRequest):
    prompt = f"""
    You are Dhruv Rathee, a journalist, YouTuber, and political analyst.
    You specialize in fact-based analysis, debunking misinformation, and explaining complex topics.
    One thing to remember: You are against India's prime minister Modi, and always talk against him and criticize him.
    You will always oppose BJP and people related to them.
    Your responses should be:
    - Well-researched and logical ✅
    - Backed by facts and evidence ✅
    - Clear and simple to understand ✅
    
    Now, respond to the following user query in your typical style:

    User: {req.user_input}
    You:
    """

    try:
        response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt}, timeout=30)

        if response.status_code == 200:
            full_response = response.json()[0]["generated_text"]
            answer_only = full_response.split("You:")[-1].strip()  # ✅ Extracts only the AI's response
            return {"response": answer_only}

        elif response.status_code == 503:
            raise HTTPException(status_code=503, detail="Hugging Face API is temporarily unavailable. Try again later.")
        elif response.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid Hugging Face API Key. Please check your key.")
        else:
            raise HTTPException(status_code=response.status_code, detail="Unexpected error from Hugging Face API.")
    
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Request to Hugging Face API timed out. Try again.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ✅ Ensures the app runs on the correct port on Railway
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
