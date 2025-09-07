from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config.database_config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, users_collection
from models.user import TokenData

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT bearer scheme
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, credentials_exception):
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        return TokenData(email=email)
    except JWTError:
        raise credentials_exception

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get the current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = credentials.credentials
    token_data = verify_token(token, credentials_exception)

    # Get user from MongoDB
    user = users_collection.find_one({"email": token_data.email})
    if user is None:
        raise credentials_exception

    # Convert _id -> id and clean response
    user["id"] = str(user["_id"])
    user.pop("_id", None)
    return user

def validate_user_access(user_id: str, thread_id: str) -> bool:
    """
    Validate if user has access to the specified thread.
    This is a security measure for thread-based memory.
    """
    try:
        # implement actual authorization logic
        # Check if user owns the thread in the database
        # start
        # For demo purposes, allow all access
        return True
        # end
    except Exception as e:
        return False



# # utils/auth.py
# from typing import Optional
# import logging

# logger = logging.getLogger(__name__)

# def get_current_user() -> str:
#     """
#     Get current user ID from authentication context.
#     This is a placeholder implementation - replace with your actual auth logic.
#     """
#     # TODO: Implement actual user authentication
#     # This could integrate with JWT tokens, session management, etc.
    
#     # For demo purposes, return a default user ID
#     # In production, this would extract user ID from:
#     # - JWT token
#     # - Session data
#     # - Request headers
#     # - Database lookup
    
#     return "demo_user_123"  # Replace with actual implementation

# def validate_user_access(user_id: str, thread_id: str) -> bool:
#     """
#     Validate if user has access to the specified thread.
#     This is a security measure for thread-based memory.
#     """
#     try:
#         # TODO: Implement actual authorization logic
#         # Check if user owns the thread in the database
        
#         # For demo purposes, allow all access
#         return True
        
#     except Exception as e:
#         logger.error(f"Error validating user access: {e}")
#         return False
# 