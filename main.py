from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config.database_config import client, ALLOWED_ORIGINS
from routes import auth, chat
from routes import agent as agent_routes


app = FastAPI()



@app.get("/")
def root():
    return {"message": "AI Blog Generator API is running!"}


# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(agent_routes.router)



