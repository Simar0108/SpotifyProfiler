#!/usr/bin/env python3
"""
Cron Setup Helper
Helps you set up the cron job to run the collector every 12 hours
"""

import os
import subprocess
from pathlib import Path

def get_project_path():
    """Get the absolute path to the project directory"""
    return Path(__file__).parent.absolute()

def get_python_path():
    """Get the path to the Python executable in the virtual environment"""
    venv_python = get_project_path() / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    else:
        # Fallback to system python
        return "python3"

def setup_cron():
    """Set up the cron job"""
    project_path = get_project_path()
    python_path = get_python_path()
    script_path = project_path / "scripts" / "robust_collector.py"
    
    print("🕐 Setting up Cron Job for Spotify Data Collection")
    print("=" * 60)
    print(f"Project path: {project_path}")
    print(f"Python path: {python_path}")
    print(f"Script path: {script_path}")
    print()
    
    # Create the cron command
    cron_command = f"0 */12 * * * cd {project_path} && PYTHONPATH=. {python_path} {script_path}"
    
    print("📋 Cron command to add:")
    print(f"   {cron_command}")
    print()
    
    print("🔧 To add this to your crontab:")
    print("   1. Run: crontab -e")
    print("   2. Add the line above")
    print("   3. Save and exit")
    print()
    
    print("📝 What this does:")
    print("   - Runs every 12 hours (at 00:00 and 12:00)")
    print("   - Changes to your project directory")
    print("   - Sets PYTHONPATH for imports")
    print("   - Runs the robust collector script")
    print()
    
    print("✅ The script will:")
    print("   - Log everything to logs/spotify_collector_YYYYMMDD.log")
    print("   - Handle errors gracefully")
    print("   - Skip duplicate tracks")
    print("   - Exit with proper status codes")
    
    return cron_command

def check_cron_status():
    """Check if the cron job is already set up"""
    try:
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        if result.returncode == 0:
            cron_content = result.stdout
            if 'robust_collector.py' in cron_content:
                print("✅ Cron job is already set up!")
                return True
            else:
                print("❌ Cron job not found in crontab")
                return False
        else:
            print("❌ No crontab found")
            return False
    except Exception as e:
        print(f"❌ Error checking crontab: {e}")
        return False

if __name__ == "__main__":
    print("🎵 Spotify Data Collector - Cron Setup")
    print("=" * 60)
    
    # Check if already set up
    if check_cron_status():
        print("\nTo modify the cron job, run: crontab -e")
    else:
        # Show setup instructions
        setup_cron() 