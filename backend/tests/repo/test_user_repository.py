import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, UserORM
from app.domain.models import User
from app.repositories.implementation.sql_alchemy_user_repo import SQLAlchemyUserRepo
from app.exceptions.repository import (
    EntityNotFoundException,
    DuplicateEntityException,
    RepositoryOperationException
)


@pytest.fixture
def session():
    """Create a fresh in-memory SQLite database for each test."""
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()


@pytest.fixture
def user_repo(session):
    """Create a user repository instance."""
    return SQLAlchemyUserRepo(session)


@pytest.fixture
def test_users(session):
    """Create test users in the database."""
    now = datetime.now()
    users = [
        UserORM(
            id=1,
            username="alice",
            email="alice@test.com",
            hashed_password="hashed_pw1",
            is_active=True,
            is_admin=False,
            is_test=False,
            created_at=now
        ),
        UserORM(
            id=2,
            username="bob",
            email="bob@test.com",
            hashed_password="hashed_pw2",
            is_active=True,
            is_admin=True,
            is_test=False,
            created_at=now
        )
    ]
    session.add_all(users)
    session.commit()
    return users


def test_get_by_id_existing_user(user_repo, test_users):
    """Test retrieving an existing user by ID."""
    user = user_repo.get_by_id(1)
    assert user is not None
    assert user.id == 1
    assert user.username == "alice"
    assert user.email == "alice@test.com"
    assert user.hashed_password == "hashed_pw1"
    assert user.is_active is True
    assert user.is_admin is False
    assert user.is_test is False


def test_get_by_id_non_existent_user(user_repo):
    """Test retrieving a non-existent user by ID."""
    user = user_repo.get_by_id(999)
    assert user is None


def test_get_by_username_existing_user(user_repo, test_users):
    """Test retrieving an existing user by username."""
    user = user_repo.get_by_username("alice")
    assert user is not None
    assert user.username == "alice"


def test_get_by_username_non_existent_user(user_repo):
    """Test retrieving a non-existent user by username."""
    user = user_repo.get_by_username("nonexistent")
    assert user is None


def test_get_by_email_existing_user(user_repo, test_users):
    """Test retrieving an existing user by email."""
    user = user_repo.get_by_email("alice@test.com")
    assert user is not None
    assert user.email == "alice@test.com"


def test_get_by_email_non_existent_user(user_repo):
    """Test retrieving a non-existent user by email."""
    user = user_repo.get_by_email("nonexistent@test.com")
    assert user is None


def test_create_user(user_repo):
    """Test creating a new user."""
    new_user = User(
        username="carol",
        email="carol@test.com",
        hashed_password="hashed_pw3",
        is_active=True,
        is_admin=False,
        is_test=False,
        created_at=datetime.now()
    )
    created_user = user_repo.create(new_user)
    assert created_user.username == "carol"
    assert created_user.email == "carol@test.com"
    assert created_user.id is not None


def test_create_duplicate_username(user_repo, test_users):
    """Test creating a user with duplicate username raises exception."""
    duplicate_user = User(
        username="alice",  # Duplicate username
        email="unique@test.com",
        hashed_password="hashed_pw",
        is_active=True
    )
    with pytest.raises(DuplicateEntityException):
        user_repo.create(duplicate_user)


def test_update_user(user_repo, test_users):
    """Test updating an existing user."""
    user = user_repo.get_by_id(1)
    user.email = "alice.new@test.com"
    user.is_admin = True
    
    updated_user = user_repo.update(user)
    assert updated_user.email == "alice.new@test.com"
    assert updated_user.is_admin is True


def test_update_non_existent_user(user_repo):
    """Test updating a non-existent user raises exception."""
    non_existent_user = User(
        id=999,
        username="nonexistent",
        email="nonexistent@test.com",
        hashed_password="hashed_pw"
    )
    with pytest.raises(EntityNotFoundException):
        user_repo.update(non_existent_user)


def test_delete_existing_user(user_repo, test_users):
    """Test deleting an existing user."""
    success = user_repo.delete(1)
    assert success is True
    assert user_repo.get_by_id(1) is None


def test_delete_non_existent_user(user_repo):
    """Test deleting a non-existent user returns False."""
    success = user_repo.delete(999)
    assert success is False


def test_get_all_users(user_repo, test_users):
    """Test retrieving all users."""
    users = user_repo.get_all()
    assert len(users) == 2
    assert any(u.username == "alice" for u in users)
    assert any(u.username == "bob" for u in users)


def test_domain_model_conversion(user_repo, test_users):
    """Test that users are correctly converted to domain models."""
    user = user_repo.get_by_id(1)
    assert user.__class__.__name__ == "User"  # Verify it's a domain model
    assert isinstance(user.id, int)
    assert isinstance(user.username, str)
    assert isinstance(user.email, str)
    assert isinstance(user.is_active, bool)
    assert isinstance(user.is_admin, bool)
    assert isinstance(user.is_test, bool) 