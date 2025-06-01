from abc import ABC, abstractmethod
from typing import List, Optional

from app.domain.models import User


class UserRepository(ABC):
    @abstractmethod
    def get_by_id(self, user_id: int) -> Optional["User"]:
        pass

    @abstractmethod
    def get_by_username(self, username: str) -> Optional["User"]:
        pass

    @abstractmethod
    def create(self, username: str, password_hash: str) -> "User":
        pass 
    
    @abstractmethod
    def get_by_email(self, email: str) -> Optional["User"]:
        pass 
    
    @abstractmethod
    def update(self, user: "User") -> "User":
        pass 
    
    @abstractmethod
    def delete(self, user_id: int) -> bool:
      pass 
    
    @abstractmethod
    def get_all(self) -> List["User"]:
        pass 