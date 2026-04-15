# backend/routes/auth.py
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

# ── Request / Response models ────────────────────────
class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword123",
                "full_name": "John Doe"
            }
        }

class LoginRequest(BaseModel):
    email: str
    password: str

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword123"
            }
        }

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_email: str

# ── Register endpoint ────────────────────────────────
@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    """
    Create a new user account.
    Full implementation comes in Step 9 with database.
    """
    # Placeholder — real logic added in Step 9
    return {
        "message": "Registration endpoint ready",
        "email": request.email,
        "status": "Full auth system coming in Step 9"
    }

# ── Login endpoint ───────────────────────────────────
@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Login with email and password.
    Returns a JWT token to use in future requests.
    """
    # Placeholder — real JWT logic added in Step 9
    return {
        "access_token": "placeholder-token-step9-will-make-this-real",
        "token_type": "bearer",
        "user_email": request.email
    }

# ── Profile endpoint ─────────────────────────────────
@router.get("/profile")
async def get_profile():
    """
    Returns current logged-in user's profile.
    Requires JWT token in Step 9.
    """
    return {
        "message": "Profile endpoint ready",
        "status": "JWT protection added in Step 9"
    }