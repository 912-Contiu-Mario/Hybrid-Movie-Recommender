
class RecommenderException(Exception):
    """Base exception for all recommender-related errors."""
    pass

class ModelNotLoadedException(RecommenderException):
    """Raised when a model or embeddings are not loaded."""
    pass

class ResourceNotFoundException(RecommenderException):
    """Raised when a required resource (file, user, item, etc.) is not found."""
    pass

class InvalidRequestException(RecommenderException):
    """Raised when an invalid request is made (e.g., invalid parameters)."""
    pass

class RecommendationFailedException(RecommenderException):
    """Raised when recommendation generation fails for any reason not covered by other exceptions."""
    pass

class ConfigurationException(RecommenderException):
    """Raised when there are issues with recommender configuration (e.g., invalid parameters, missing required settings)."""
    pass
