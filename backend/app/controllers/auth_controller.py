from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.domain.dto import UserCreate, UserResponse, Token
from app.service.dependencies import get_auth_service
from app.service.auth_service import AuthService
from app.exceptions.auth import UserAlreadyExistsException, InvalidCredentialsException

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
    responses={404: {"description": "Not found"}}
)

@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
def register(
    user_data: UserCreate,
    auth_service: AuthService = Depends(get_auth_service)
):
    try:
        user = auth_service.register_user(user_data.username, user_data.email, user_data.password)
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email
        )
    except UserAlreadyExistsException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user"
        )

@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service)
):
    try:
        _, access_token = auth_service.authenticate_user(form_data.username, form_data.password)
        return Token(access_token=access_token, token_type="bearer")
    except InvalidCredentialsException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to login"
        )
