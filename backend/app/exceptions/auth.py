class AuthException(Exception):
    """Base exception for authentication errors"""
    pass

class UserAlreadyExistsException(AuthException):
    """Raised when attempting to register a user with an existing username or email"""
    pass

class InvalidCredentialsException(AuthException):
    """Raised when login credentials are invalid"""
    pass 