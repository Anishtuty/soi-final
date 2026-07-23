import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _resolve_path(value: str, default: Path) -> str:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path)


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
    UPLOAD_FOLDER = _resolve_path(os.getenv("UPLOAD_FOLDER", str(BASE_DIR / "uploads")), BASE_DIR / "uploads")
    MODEL_PATH = _resolve_path(os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "snake_model.h5")), BASE_DIR / "models" / "snake_model.h5")
    DATA_PATH = _resolve_path(os.getenv("DATA_PATH", str(BASE_DIR / "data" / "snakes.json")), BASE_DIR / "data" / "snakes.json")
    SNAKES_JSON = DATA_PATH  # Alias used by app.py routes
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
