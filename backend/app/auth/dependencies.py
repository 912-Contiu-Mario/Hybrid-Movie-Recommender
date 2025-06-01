from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.config.environment import JWT_SECRET_KEY, JWT_ALGORITHM
from app.domain.dto import TokenData
from app.db.database import get_db
from app.repositories import SQLAlchemyUserRepo

# OAuth2 scheme for FastAPI
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        # create token data
        token_data = TokenData(user_id=user_id)
    except JWTError:
        raise credentials_exception
    
    # get user from database
    user_repo = SQLAlchemyUserRepo(db)
    user = user_repo.get_by_id(token_data.user_id)
    if user is None:
        raise credentials_exception
        
    return user

def get_current_active_user(current_user = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user 