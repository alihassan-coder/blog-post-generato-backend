from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class Message(BaseModel):
    id: Optional[str] = None   # store ObjectId as string
    content: str
    is_user: bool
    timestamp: Optional[datetime] = None
    chat_id: Optional[str] = None


class MessageCreate(BaseModel):
    content: str
    is_user: bool


class Chat(BaseModel):
    id: Optional[str] = None   # store ObjectId as string
    title: str
    user_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    messages: List[Message] = []


class ChatCreate(BaseModel):
    title: str


class ChatUpdate(BaseModel):
    title: Optional[str] = None
