from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)

db = client["bloge_db"]

# Define collection

users_collection = db["users"]  # Store registered users
blogs_collection = db["blogs"]  # Store generated blogs
tokens_collection = db["tokens"]  # Optional: blacklist/refresh tokens
logs_collection = db["logs"]  # API request/response logs
chats_collection= db["chats"]  # Store user chats
messages_collection = db["messages"]  # Store chat messages

ALLOWED_ORIGINS = '*'  # Allow all origins for simplicity; adjust in production
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))