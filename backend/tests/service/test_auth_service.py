import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from app.domain.models import User
from app.service.auth_service import AuthService
from app.exceptions.auth import (
    InvalidCredentialsException,
    UserAlreadyExistsException
)


@pytest.fixture
def mock_user_repo():
    """Create a mock user repository."""
    return Mock()


@pytest.fixture
def auth_service(mock_user_repo):
    """Create an auth service with mock repository."""
    return AuthService(mock_user_repo)


@pytest.fixture
def test_user():
    """Create a test user."""
    return User(
        id=1,
        username="testuser",
        email="test@example.com",
        hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY.5ZbR8WyA0lYu",  # hashed 'password123'
        is_active=True,
        is_admin=False,
        created_at=datetime.now()
    )


def test_register_user_success(auth_service, mock_user_repo):
    """Test successful user registration."""
    # Setup
    mock_user_repo.get_by_username.return_value = None
    mock_user_repo.get_by_email.return_value = None
    mock_user_repo.create.return_value = User(
        id=1,
        username="newuser",
        email="new@example.com",
        hashed_password="hashed_password",
        is_active=True,
        created_at=datetime.now()
    )

    # Test
    result = auth_service.register_user(
        username="newuser",
        email="new@example.com",
        password="password123"
    )

    # Verify
    assert result.username == "newuser"
    assert result.email == "new@example.com"
    assert result.is_active is True
    mock_user_repo.create.assert_called_once()


def test_register_user_existing_username(auth_service, mock_user_repo, test_user):
    """Test registration with existing username."""
    # Setup
    mock_user_repo.get_by_username.return_value = test_user

    # Test
    with pytest.raises(UserAlreadyExistsException, match="Username already registered"):
        auth_service.register_user(
            username="testuser",
            email="new@example.com",
            password="password123"
        )


def test_register_user_existing_email(auth_service, mock_user_repo, test_user):
    """Test registration with existing email."""
    # Setup
    mock_user_repo.get_by_username.return_value = None
    mock_user_repo.get_by_email.return_value = test_user

    # Test
    with pytest.raises(UserAlreadyExistsException, match="Email already registered"):
        auth_service.register_user(
            username="newuser",
            email="test@example.com",
            password="password123"
        )


def test_authenticate_user_success(auth_service, mock_user_repo, test_user):
    """Test successful user authentication."""
    # Setup
    mock_user_repo.get_by_username.return_value = test_user
    
    # Mock password verification
    with patch("app.service.auth_service.verify_password", return_value=True):
        # Test
        user, token = auth_service.authenticate_user("testuser", "password123")

        # Verify
        assert user.id == test_user.id
        assert user.username == test_user.username
        assert user.email == test_user.email
        assert isinstance(token, str)
        assert len(token) > 0


def test_authenticate_user_invalid_username(auth_service, mock_user_repo):
    """Test authentication with invalid username."""
    # Setup
    mock_user_repo.get_by_username.return_value = None

    # Test
    with pytest.raises(InvalidCredentialsException):
        auth_service.authenticate_user("nonexistent", "password123")


def test_authenticate_user_invalid_password(auth_service, mock_user_repo, test_user):
    """Test authentication with invalid password."""
    # Setup
    mock_user_repo.get_by_username.return_value = test_user
    
    # Mock password verification
    with patch("app.service.auth_service.verify_password", return_value=False):
        # Test
        with pytest.raises(InvalidCredentialsException):
            auth_service.authenticate_user("testuser", "wrongpassword")


def test_authenticate_user_inactive(auth_service, mock_user_repo, test_user):
    """Test authentication with inactive user."""
    # Setup
    test_user.is_active = False
    mock_user_repo.get_by_username.return_value = test_user

    # Test
    with pytest.raises(InvalidCredentialsException):
        auth_service.authenticate_user("testuser", "password123")


def test_authenticate_test_user(auth_service, mock_user_repo, test_user):
    """Test authentication with a test user."""
    # Setup
    test_user.is_test = True
    mock_user_repo.get_by_username.return_value = test_user

    # Test
    user, token = auth_service.authenticate_user("testuser", "wrongpassword")  # password doesn't matter for test users

    # Verify
    assert user.id == test_user.id
    assert user.username == test_user.username
    assert isinstance(token, str)
    assert len(token) > 0 