class RatingServiceException(Exception):
    """Base exception for service operation errors."""
    pass

class ResourceNotFoundException(RatingServiceException):
    """Raised when a requested resource is not found."""
    pass

class InvalidRequestException(RatingServiceException):
    """Raised when request parameters are invalid (e.g. user/movie not found, invalid rating value)."""
    pass