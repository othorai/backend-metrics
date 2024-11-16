#routers/users.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.models.models import User, InteractionHistory, LikedPost, Article, Organization, user_organizations
from app.schemas.schemas import UserCreate, User as UserSchema, Token, ChatHistoryResponse, Organization as OrganizationSchema, EmailRequest, UserResponse
from app.utils.database import get_db
from app.utils.auth import get_password_hash, verify_password, create_access_token, get_current_user
from app.services.email_service import send_welcome_email
import logging
from typing import List
from app.schemas.schemas import LikedPostResponse
from sqlalchemy.exc import IntegrityError
import uuid

router = APIRouter()

def get_full_data_access():
    return "all_departments,all_locations,all_products,financial_data,customer_data,marketing_data,sales_data,employee_data,historical_data,forecasts,system_config,audit_logs"

@router.post("/signup", response_model=UserSchema)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    username_exists = db.query(User).filter(User.username == user.username).first()
    if username_exists:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    hashed_password = get_password_hash(user.password)
    
    # Determine data access
    if user.data_access is None or user.data_access.lower() in ["full", "all", "everything"]:
        data_access = get_full_data_access()
    else:
        data_access = user.data_access
    
    # Check if organization exists, if not, create it
    org = db.query(Organization).filter(Organization.name == user.organization_name).first()
    if not org:
        org = Organization(name=user.organization_name)
        db.add(org)
        db.commit()
        db.refresh(org)
    
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        role=user.role,
        data_access=data_access,
        is_active=True,
        is_admin=user.role.lower() in ["admin", "ceo"]
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Add user to organization
    db_user.organizations.append(org)
    db.commit()

    send_welcome_email(db_user.email)
    return db_user

@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    logging.debug(f"Login attempt for email: {form_data.username}")
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user:
        logging.debug(f"User not found for email: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not verify_password(form_data.password, user.hashed_password):
        logging.debug(f"Invalid password for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user's organizations
    if not user.organizations:
        raise HTTPException(status_code=400, detail="User is not associated with any organization")
    
    # For simplicity, we're using the first organization. In a real-world scenario,
    # you might want to let the user choose which organization to log into
    
    access_token = create_access_token(
        data={"sub": user.email, "org_id": user.organizations[0].id if user.organizations else None}
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/user/organizations", response_model=List[OrganizationSchema])
def get_user_organizations(current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    user = current_user["user"]
    return user.organizations

@router.post("/switch-organization/{org_id}", response_model=Token)
def switch_organization(
    org_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user = current_user["user"]
    org = db.query(Organization).filter(Organization.id == org_id).first()
    
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    if org not in user.organizations:
        raise HTTPException(status_code=403, detail=f"User is not a member of organization {org_id}")
    
    access_token = create_access_token(data={"sub": user.email, "org_id": org_id})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/user/{user_id}/add-organization/{org_id}")
def add_user_to_organization(
    user_id: int,
    org_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    if not current_user["user"].is_admin:
        raise HTTPException(status_code=403, detail="Only admins can add users to organizations")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    org = db.query(Organization).filter(Organization.id == org_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    if org not in user.organizations:
        user.organizations.append(org)
        db.commit()
        return {"message": f"User {user.username} added to organization {org.name}"}
    else:
        return {"message": f"User {user.username} is already a member of organization {org.name}"}
    
@router.get("/me", response_model=UserSchema)
def get_current_user_info(current_user: dict = Depends(get_current_user)):
    return current_user["user"]

@router.get("/chat-history/{session_id}", response_model=List[ChatHistoryResponse])
def get_chat_history(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat_history = db.query(InteractionHistory).filter(
        InteractionHistory.user_id == current_user["user"].id,
        InteractionHistory.session_id == session_id
    ).order_by(InteractionHistory.timestamp).all()

    return [
        ChatHistoryResponse(
            question=interaction.question,
            answer=interaction.answer,
            timestamp=interaction.timestamp
        ) for interaction in chat_history
    ]

@router.post("/like/{article_id}", response_model=LikedPostResponse)
def like_post(
    article_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    try:
        liked_post = LikedPost(user_id=current_user["user"].id, article_id=article_id)
        db.add(liked_post)
        db.commit()
        db.refresh(liked_post)
        return LikedPostResponse(message="Post liked successfully", liked=True)
    except IntegrityError:
        db.rollback()
        return LikedPostResponse(message="Post already liked", liked=True)

@router.delete("/unlike/{article_id}", response_model=LikedPostResponse)
def unlike_post(
    article_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    liked_post = db.query(LikedPost).filter(
        LikedPost.user_id == current_user["user"].id,
        LikedPost.article_id == article_id
    ).first()

    if not liked_post:
        return LikedPostResponse(message="Post was not liked", liked=False)

    db.delete(liked_post)
    db.commit()
    return LikedPostResponse(message="Post unliked successfully", liked=False)

@router.get("/liked-posts", response_model=List[uuid.UUID])
def get_liked_posts(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    liked_posts = db.query(LikedPost.article_id).filter(LikedPost.user_id == current_user["user"].id).all()
    return [post.article_id for post in liked_posts]

@router.post("/find-by-email", response_model=UserResponse)
def find_user_by_email(
    email_data: EmailRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    if not current_user["user"].is_admin:
        raise HTTPException(status_code=403, detail="Only admins can look up users")
    
    user = db.query(User).filter(User.email == email_data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user