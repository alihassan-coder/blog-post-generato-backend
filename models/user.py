from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


# Base user model
class UserBase(BaseModel):
    email: EmailStr
    name: str


# For user registration
class UserCreate(UserBase):
    password: str


# For user login
class UserLogin(BaseModel):
    email: EmailStr
    password: str


# For returning user data (without password)
class User(UserBase):
    id: Optional[str] = None     # store ObjectId as string
    created_at: Optional[datetime] = None
    is_active: bool = True


# For storing user in DB (with hashed password)
class UserInDB(User):
    hashed_password: str


# For JWT token response
class Token(BaseModel):
    access_token: str
    token_type: str


# For token payload
class TokenData(BaseModel):
    email: Optional[str] = None
