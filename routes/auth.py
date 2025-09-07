from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer
from datetime import timedelta, datetime
from bson import ObjectId

from config.database_config import users_collection, ACCESS_TOKEN_EXPIRE_MINUTES
from models.user import UserCreate, UserLogin, User, Token
from utils.auth import get_password_hash, verify_password, create_access_token, get_current_user

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", response_model=User)
def register(user_data: UserCreate):
    """Register a new user."""
    # Check if user already exists
    existing_user = users_collection.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash password and create user
    hashed_password = get_password_hash(user_data.password)
    user_dict = {
        "email": user_data.email,
        "name": user_data.name,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow(),
        "is_active": True
    }
    
    result = users_collection.insert_one(user_dict)
    user_dict["id"] = str(result.inserted_id)  # Convert ObjectId to str
    
    # Remove password from response
    user_dict.pop("hashed_password", None)
    
    return User(**user_dict)

@router.post("/login", response_model=Token)
def login(user_credentials: UserLogin):
    """Login user and return JWT token."""
    # Find user by email
    user = users_collection.find_one({"email": user_credentials.email})
    if not user or not verify_password(user_credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=User)
def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information."""
    current_user.pop("hashed_password", None)
    return User(**current_user)

@router.post("/logout")
def logout():
    """Logout user (client should remove token)."""
    return {"message": "Successfully logged out"}
