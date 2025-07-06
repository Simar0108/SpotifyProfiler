from database.database import SessionLocal
from database.simple_models import ListeningHistory
from sqlalchemy import func

def view_simple_data():
    """View the simplified data we've collected"""
    db = SessionLocal()
    
    try:
        # Get total records
        total_records = db.query(ListeningHistory).count()
        
        print(f"📊 Simple Data Overview")
        print("=" * 50)
        print(f"Total listening records: {total_records}")
        print()
        
        if total_records > 0:
            # Get recent listening history
            recent_listens = db.query(ListeningHistory).order_by(
                ListeningHistory.played_at.desc()
            ).limit(10).all()
            
            print(f"🎵 Recent Listening History (last 10):")
            for listen in recent_listens:
                print(f"   {listen.played_at.strftime('%Y-%m-%d %H:%M')} ({listen.day_of_week}, {listen.time_of_day})")
                print(f"     {listen.track_name} - {listen.artist_name}")
                print()
            
            # Get time-of-day breakdown
            time_breakdown = db.query(
                ListeningHistory.time_of_day,
                func.count(ListeningHistory.id).label('count')
            ).group_by(ListeningHistory.time_of_day).all()
            
            print(f"⏰ Time-of-Day Breakdown:")
            for time_period, count in time_breakdown:
                print(f"   {time_period}: {count} tracks")
            print()
            
            # Get day-of-week breakdown
            day_breakdown = db.query(
                ListeningHistory.day_of_week,
                func.count(ListeningHistory.id).label('count')
            ).group_by(ListeningHistory.day_of_week).all()
            
            print(f"📅 Day-of-Week Breakdown:")
            for day_name, count in day_breakdown:
                print(f"   {day_name}: {count} tracks")
            print()
            
            # Get top artists
            top_artists = db.query(
                ListeningHistory.artist_name,
                func.count(ListeningHistory.id).label('count')
            ).group_by(ListeningHistory.artist_name).order_by(
                func.count(ListeningHistory.id).desc()
            ).limit(5).all()
            
            print(f"🎤 Top Artists:")
            for artist, count in top_artists:
                print(f"   {artist}: {count} tracks")
        
    except Exception as e:
        print(f"Error viewing data: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    view_simple_data() 