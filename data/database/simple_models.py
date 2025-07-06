from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.sql import func
from database.database import Base
import uuid

class ListeningHistory(Base):
    __tablename__ = "listening_history"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    track_name = Column(Text, nullable=False)
    artist_name = Column(Text, nullable=False)
    spotify_id = Column(Text, nullable=False, index=True)
    played_at = Column(DateTime, nullable=False, index=True)
    local_time = Column(DateTime)  # Optional conversion to local time
    day_of_week = Column(Text)  # e.g. "Monday"
    time_of_day = Column(Text)  # "Morning", "Afternoon", "Night"
    created_at = Column(DateTime, default=func.now()) 