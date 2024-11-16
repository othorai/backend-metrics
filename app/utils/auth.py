#auth.py
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.utils.database import get_db  # Change from database import get_db
from app.models.models import User, Organization, user_organizations
from app.utils.config import settings
from app.schemas.schemas import UserCreate

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="authorization/login")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

import logging

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        logging.info(f"Decoding token: {token}")
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        logging.info(f"Decoded payload: {payload}")
        email: str = payload.get("sub")
        if email is None:
            logging.error("Email not found in token payload")
            raise credentials_exception
        org_id: int = payload.get("org_id")
        logging.info(f"Email: {email}, Org ID: {org_id}")
    except JWTError as e:
        logging.error(f"JWT decode error: {str(e)}")
        raise credentials_exception
    
    user = db.query(User).filter((User.email == email) | (User.username == email)).first()
    if user is None:
        logging.error(f"User not found: {email}")
        raise credentials_exception
    logging.info(f"User authenticated: {user.id}")
    return {"user": user, "current_org_id": org_id}

async def get_current_active_user(current_user: dict = Depends(get_current_user)):
    if not current_user["user"].is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_organization(current_user: dict = Depends(get_current_active_user), db: Session = Depends(get_db)):
    org = db.query(Organization).filter(Organization.id == current_user["current_org_id"]).first()
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org

def create_user(db: Session, user: UserCreate, organization_id: int):
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        role=user.role,
        data_access=user.data_access
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Add user to organization
    org = db.query(Organization).filter(Organization.id == organization_id).first()
    if org:
        db_user.organizations.append(org)
        db.commit()

    return db_user

def authenticate_user(db: Session, email: str, password: str):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def get_user_organizations(db: Session, user_id: int):
    return db.query(Organization).join(user_organizations).filter(user_organizations.c.user_id == user_id).all()