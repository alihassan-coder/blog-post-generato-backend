from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
from bson import ObjectId 
from datetime import datetime
from config.database_config import chats_collection, messages_collection
from models.chat import Chat, ChatCreate, ChatUpdate, Message, MessageCreate
from utils.auth import get_current_user

router = APIRouter(prefix="/chat", tags=["chat"])

from bson import ObjectId

def serialize_chat(chat: dict) -> dict:
    """Convert MongoDB chat document to JSON-serializable dict."""
    chat["id"] = str(chat["_id"])
    chat.pop("_id", None)

    if "user_id" in chat and isinstance(chat["user_id"], ObjectId):
        chat["user_id"] = str(chat["user_id"])

    if "messages" in chat:
        for msg in chat["messages"]:
            if "id" in msg and isinstance(msg["id"], ObjectId):
                msg["id"] = str(msg["id"])
            if "chat_id" in msg and isinstance(msg["chat_id"], ObjectId):
                msg["chat_id"] = str(msg["chat_id"])
            # also ensure chat_id and id are strings if they are already strings
            if "id" in msg and isinstance(msg["id"], str):
                msg["id"] = msg["id"]
            if "chat_id" in msg and isinstance(msg["chat_id"], str):
                msg["chat_id"] = msg["chat_id"]

    return chat



@router.post("/", response_model=Chat)
def create_chat(chat_data: ChatCreate, current_user: dict = Depends(get_current_user)):
    """Create a new chat."""
    chat_dict = {
        "title": chat_data.title,
        "user_id": ObjectId(current_user["id"]),  # always use logged-in user
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "messages": []
    }

    result = chats_collection.insert_one(chat_dict)
    chat_dict["id"] = str(result.inserted_id)

    # ensure returned chat is JSON-serializable
    return Chat(**serialize_chat(chat_dict))


@router.get("/", response_model=List[Chat])
def get_chats(current_user: dict = Depends(get_current_user)):
    """Get all chats for the current user."""
    chats = []
    for chat in chats_collection.find({"user_id": ObjectId(current_user["id"])}).sort("updated_at", -1):
        chat = serialize_chat(chat)
        chats.append(Chat(**chat))
    return chats





@router.get("/{chat_id}", response_model=Chat)
def get_chat(chat_id: str, current_user: dict = Depends(get_current_user)):
    """Get a specific chat by ID."""
    if not ObjectId.is_valid(chat_id):
        raise HTTPException(status_code=400, detail="Invalid chat ID")

    chat = chats_collection.find_one({
        "_id": ObjectId(chat_id),
        "user_id": ObjectId(current_user["id"])
    })

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    chat = serialize_chat(chat)
    return Chat(**chat)


@router.put("/{chat_id}", response_model=Chat)
def update_chat(chat_id: str, chat_update: ChatUpdate, current_user: dict = Depends(get_current_user)):
    """Update a chat."""
    if not ObjectId.is_valid(chat_id):
        raise HTTPException(status_code=400, detail="Invalid chat ID")

    update_data = {"updated_at": datetime.utcnow()}
    if chat_update.title is not None:
        update_data["title"] = chat_update.title

    result = chats_collection.update_one(
        {"_id": ObjectId(chat_id), "user_id": ObjectId(current_user["id"])},
        {"$set": update_data}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found")

    chat = chats_collection.find_one({"_id": ObjectId(chat_id)})
    chat = serialize_chat(chat)
    return Chat(**chat)


@router.delete("/{chat_id}")
def delete_chat(chat_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a chat and all its messages."""
    if not ObjectId.is_valid(chat_id):
        raise HTTPException(status_code=400, detail="Invalid chat ID")

    chat_result = chats_collection.delete_one({
        "_id": ObjectId(chat_id),
        "user_id": ObjectId(current_user["id"])
    })

    if chat_result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found")

    messages_collection.delete_many({"chat_id": ObjectId(chat_id)})

    return {"message": "Chat deleted successfully"}


@router.post("/{chat_id}/messages", response_model=Message)
def create_message(chat_id: str, message_data: MessageCreate, current_user: dict = Depends(get_current_user)):
    """Create a new message in a chat."""
    if not ObjectId.is_valid(chat_id):
        raise HTTPException(status_code=400, detail="Invalid chat ID")

    chat = chats_collection.find_one({
        "_id": ObjectId(chat_id),
        "user_id": ObjectId(current_user["id"])
    })

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Prepare DB object with ObjectId, but return a serializable response
    db_message = {
        "content": message_data.content,
        "is_user": message_data.is_user,
        "chat_id": ObjectId(chat_id),
        "timestamp": datetime.utcnow()
    }

    result = messages_collection.insert_one(db_message)

    # Update chat timestamp
    chats_collection.update_one(
        {"_id": ObjectId(chat_id)},
        {"$set": {"updated_at": datetime.utcnow()}}
    )

    # Build response message with string ids
    resp_message = {
        "id": str(result.inserted_id),
        "content": db_message["content"],
        "is_user": db_message["is_user"],
        "chat_id": str(chat_id),
        "timestamp": db_message["timestamp"]
    }

    return Message(**resp_message)


@router.get("/{chat_id}/messages", response_model=List[Message])
def get_chat_messages(chat_id: str, current_user: dict = Depends(get_current_user)):
    """Get all messages in a chat."""
    if not ObjectId.is_valid(chat_id):
        raise HTTPException(status_code=400, detail="Invalid chat ID")

    chat = chats_collection.find_one({
        "_id": ObjectId(chat_id),
        "user_id": ObjectId(current_user["id"])
    })

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    messages = []
    for message in messages_collection.find({"chat_id": ObjectId(chat_id)}).sort("timestamp", 1):
        message["id"] = str(message["_id"])
        # convert chat_id to string for response
        if "chat_id" in message and isinstance(message["chat_id"], ObjectId):
            message["chat_id"] = str(message["chat_id"])
        message.pop("_id", None)
        messages.append(Message(**message))

    return messages
