import socket
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings


def resolve_db_host(url: str) -> str:
    """Traduce el host 'db' (red Docker) a 'localhost' si no resuelve.

    Dentro de Docker 'db' resuelve y se mantiene; fuera (host/tests) no
    resuelve y se usa localhost (mismas credenciales).
    """
    if "@db:" in url:
        try:
            socket.gethostbyname("db")
        except socket.gaierror:
            return url.replace("@db:", "@localhost:")
    return url


_settings = get_settings()
_database_url = resolve_db_host(_settings.database_url)
engine = create_engine(_database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, expire_on_commit=False, future=True
)


def get_session() -> Generator[Session, None, None]:
    """Dependencia de FastAPI / uso directo con `with`."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager: sesión con commit/rollback automático.

    Uso:
        with session_scope() as s:
            s.execute(text("..."))
    # commit automático al salir sin excepción; rollback si algo falla.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def run_in_own_session(fn, *args, **kwargs):
    """Ejecuta fn con una sesión propia (Session no es thread-safe).

    SQLAlchemy Session NO es thread-safe: compartir la sesión del caller
    entre hilos lanza InvalidSessionError ("session is provisioning a new
    connection") cuando dos hilos pisan la adquisición de conexión. Cada
    tarea paralela abre su propia sesión (SessionLocal) y la cierra al
    terminar. Las tareas son lecturas read-only: no hay transacción que
    propagar de vuelta al caller.
    """
    session = SessionLocal()
    try:
        return fn(session, *args, **kwargs)
    finally:
        session.close()
