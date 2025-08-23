import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Spotify API credentials
    SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
    SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/callback")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sonic_sync.db")
    
    # OpenAI API (for later use)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # App settings
    APP_NAME = "Sonic Sync"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

settings = Settings() 