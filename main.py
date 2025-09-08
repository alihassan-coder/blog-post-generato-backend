from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config.database_config import client, ALLOWED_ORIGINS
from routes import auth, chat
from routes import agent as agent_routes


# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting up Blog Generator API...")
    yield
    # Shutdown
    print("🛑 Shutting down Blog Generator API...")
    client.close()  # close MongoDB connection

# FastAPI app
app = FastAPI(
    title="AI Blog Generator API",
    description="Backend API for AI-powered blog content generation with MongoDB and JWT authentication",
    version="1.0.0",
    lifespan=lifespan,
)

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

# Health routes
@app.get("/")
def root():
    return {"message": "AI Blog Generator API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is running successfully"}

# # 👇 Required for Vercel
# handler = Mangum(app)

# # Local dev entrypoint
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
