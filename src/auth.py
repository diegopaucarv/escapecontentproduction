"""
RBAC real (lo que la propuesta original pedía y no tenía ni una tabla
de usuarios). JWT simple + bcrypt — sin passlib de por medio, una
dependencia menos, misma seguridad para este volumen de usuarios
(un equipo de pocas personas, no una app con miles de logins).

Diseño deliberado: los roles son exactamente los tres del diagrama
original (🟨 lider, 🟦 equipo, 🟩 externo) — no se inventó una
jerarquía de permisos más fina de la que el proceso de negocio ya usa.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import get_settings
from src.db.models import AppUser
from src.db.session import get_session

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # una jornada de trabajo

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(user: AppUser) -> str:
    settings = get_settings()
    payload = {
        "sub": str(user.id),
        "role": user.role.value if hasattr(user.role, "value") else user.role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def authenticate_user(session: Session, email: str, password: str) -> AppUser | None:
    user = session.scalar(select(AppUser).where(AppUser.email == email, AppUser.is_active.is_(True)))
    if user is None or user.hashed_password is None:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> AppUser:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudo validar la credencial",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        settings = get_settings()
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_error
    except jwt.PyJWTError:
        raise credentials_error

    user = session.get(AppUser, uuid.UUID(user_id))
    if user is None or not user.is_active:
        raise credentials_error
    return user


def require_role(*allowed_roles: str):
    """Dependencia paramétrica: `Depends(require_role('lider'))`.
    Los 'auto_pass' del Gatekeeper (src/agents/gatekeeper.py) nunca
    pasan por aquí — esto protege endpoints HTTP que representan
    decisiones humanas explícitas (ej. OWNER_APPROVAL)."""

    def _dependency(user: AppUser = Depends(get_current_user)) -> AppUser:
        role_value = user.role.value if hasattr(user.role, "value") else user.role
        if role_value not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Esta acción requiere rol {allowed_roles}, tienes '{role_value}'.",
            )
        return user

    return _dependency
