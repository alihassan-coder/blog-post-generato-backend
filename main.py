from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

@app.get("/")
def home():
    return {"message": "FastAPI running on Vercel 🚀"}

# Wrap FastAPI for serverless
handler = Mangum(app)
