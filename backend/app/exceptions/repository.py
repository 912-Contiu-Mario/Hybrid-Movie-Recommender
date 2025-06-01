from typing import Any

class RepositoryException(Exception):
    """Base exception for all repository-related errors."""
    pass

class EntityNotFoundException(RepositoryException):
    """Raised when an entity cannot be found in the repository."""
    pass

class DuplicateEntityException(RepositoryException):
    """Raised when attempting to create a duplicate entity."""
    pass

class InvalidEntityDataException(RepositoryException):
    """Raised when entity data is invalid or malformed."""
    pass

class RepositoryOperationException(RepositoryException):
    """Raised when a repository operation fails for any reason not covered by other exceptions."""
    pass

class ConnectionException(RepositoryException):
    """Raised when repository connection or transaction fails."""
    pass
