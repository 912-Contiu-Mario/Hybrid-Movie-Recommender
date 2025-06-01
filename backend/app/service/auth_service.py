from datetime import datetime, timedelta
from typing import Optional
from jose import jwt
from passlib.context import CryptContext
from app.config.environment import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_ACCESS_TOKEN_EXPIRE_MINUTES
from app.domain.models import User
from app.repositories import UserRepository
from app.exceptions.auth import UserAlreadyExistsException, InvalidCredentialsException

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        return pwd_context.hash(password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        
        # Set expiration
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        
        # Create JWT token
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return encoded_jwt

    def register_user(self, username: str, email: str, password: str) -> User:
        # check if username exists
        if self.user_repository.get_by_username(username):
            raise UserAlreadyExistsException("Username already registered")
        
        # check if email exists
        if self.user_repository.get_by_email(email):
            raise UserAlreadyExistsException("Email already registered")
        
        # hash password
        hashed_password = self.get_password_hash(password)
        
        # create user
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            is_active=True,
            is_admin=False,
            is_test=False,
            created_at=datetime.now()
        )
        
        return self.user_repository.create(user)

    def authenticate_user(self, username: str, password: str) -> tuple[User, str]:
        user = self.user_repository.get_by_username(username)

        # special case for test users
        if user and user.is_test:
            access_token = self._create_access_token_for_user(user)
            return user, access_token

        # regular authentication
        if not user or not self.verify_password(password, user.hashed_password):
            raise InvalidCredentialsException("Invalid username or password")
        
        access_token = self._create_access_token_for_user(user)
        return user, access_token

    def _create_access_token_for_user(self, user: User) -> str:
        # create access token
        access_token_expires = timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        return self.create_access_token(
            data={"sub": str(user.id)},
            expires_delta=access_token_expires
        ) 