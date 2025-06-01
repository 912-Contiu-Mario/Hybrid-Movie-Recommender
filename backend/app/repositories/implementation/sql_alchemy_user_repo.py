from datetime import datetime
from sqlalchemy.orm import Session
from typing import Optional, List
from sqlalchemy.exc import IntegrityError

from app.domain.models import User
from app.db.models import UserORM
from app.repositories import UserRepository
from app.exceptions.repository import (
    EntityNotFoundException,
    DuplicateEntityException,
    RepositoryOperationException,
)

class SQLAlchemyUserRepo(UserRepository):
    def __init__(self, db: Session):
        self.db = db

    def _to_domain(self, user_orm: UserORM) -> User:
        return User(
            id=user_orm.id,
            username=user_orm.username,
            email=user_orm.email,
            hashed_password=user_orm.hashed_password,
            is_active=user_orm.is_active,
            is_admin=user_orm.is_admin,
            is_test=user_orm.is_test,
            created_at=user_orm.created_at
        )
    
    def _to_orm(self, user: User) -> UserORM:
        return UserORM(
            id=user.id,
            username=user.username,
            email=user.email,
            hashed_password=user.hashed_password,
            is_active=user.is_active,
            is_admin=user.is_admin,
            is_test=user.is_test,
            created_at=user.created_at
        )
    
    def get_by_id(self, user_id: int) -> Optional[User]:
        """Get a user by ID"""
        try:
            user_orm = self.db.query(UserORM).filter(UserORM.id == user_id).first()
            return self._to_domain(user_orm) if user_orm else None
        except Exception as e:
            raise RepositoryOperationException(f"Failed to get user by ID: {str(e)}")
    
    def get_by_username(self, username: str) -> Optional[User]:
        """Get a user by username"""
        try:
            user_orm = self.db.query(UserORM).filter(UserORM.username == username).first()
            return self._to_domain(user_orm) if user_orm else None
        except Exception as e:
            raise RepositoryOperationException(f"Failed to get user by username: {str(e)}")
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get a user by email"""
        try:
            user_orm = self.db.query(UserORM).filter(UserORM.email == email).first()
            return self._to_domain(user_orm) if user_orm else None
        except Exception as e:
            raise RepositoryOperationException(f"Failed to get user by email: {str(e)}")
    
    def create(self, user: User) -> User:
        """Create a new user"""
        try:
            user_orm = self._to_orm(user)
            self.db.add(user_orm)
            self.db.commit()
            self.db.refresh(user_orm)
            return self._to_domain(user_orm)
        except IntegrityError:
            raise DuplicateEntityException("User already exists with these credentials")
        except Exception as e:
            raise RepositoryOperationException(f"Failed to create user: {str(e)}")
    
    def update(self, user: User) -> User:
        try:
            user_orm = self.db.get(UserORM, user.id)
            if not user_orm:
                raise EntityNotFoundException(f"User {user.id} not found")
            
            user_orm.username = user.username
            user_orm.email = user.email
            user_orm.hashed_password = user.hashed_password
            user_orm.is_active = user.is_active
            user_orm.is_admin = user.is_admin
            user_orm.is_test = user.is_test
            
            self.db.commit()
            self.db.refresh(user_orm)
            return self._to_domain(user_orm)
        except EntityNotFoundException:
            raise
        except IntegrityError:
            raise DuplicateEntityException("User with these credentials already exists")
        except Exception as e:
            raise RepositoryOperationException(f"Failed to update user: {str(e)}")
    
    def delete(self, user_id: int) -> bool:
        try:
            user_orm = self.db.get(UserORM, user_id)
            if not user_orm:
                return False
            
            self.db.delete(user_orm)
            self.db.commit()
            return True
        except Exception as e:
            raise RepositoryOperationException(f"Failed to delete user: {str(e)}")
    
    def get_all(self) -> List[User]:
        try:
            user_orms = self.db.query(UserORM).all()
            return [self._to_domain(user_orm) for user_orm in user_orms]
        except Exception as e:
            raise RepositoryOperationException(f"Failed to get all users: {str(e)}") 