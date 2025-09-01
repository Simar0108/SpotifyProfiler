#!/usr/bin/env python3
"""
Quick launcher for Fixed Second Round Hyperparameter Experiments

This version fixes the loss calculation issues and uses a simpler model architecture.
"""

import subprocess
import sys
import os

def main():
    """Run the fixed experiments"""
    print("🚀 Starting Fixed Second Round Experiments")
    print("=" * 50)
    
    # Check if the fixed script exists
    script_path = "pipeline/second_round_experiments_fixed.py"
    if not os.path.exists(script_path):
        print(f"❌ Fixed experiment script not found: {script_path}")
        return
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected")
        print("Please activate your virtual environment first:")
        print("   source venv/bin/activate")
        return
    
    print("✅ Virtual environment detected")
    print("✅ Fixed script found")
    print()
    print("🔧 What was fixed:")
    print("   • Simplified model architecture (removed complex hidden layers)")
    print("   • Fixed loss calculation issues")
    print("   • Added proper logging for all epochs")
    print("   • Used proven architecture from first round")
    print()
    
    # Confirm before running
    response = input("🚀 Ready to start fixed experiments? (y/N): ")
    if response.lower() != 'y':
        print("❌ Aborted by user")
        return
    
    print("🚀 Starting experiments...")
    print("   Press Ctrl+C to pause (experiments will resume from checkpoint)")
    print()
    
    try:
        # Run the fixed experiment script
        result = subprocess.run([sys.executable, script_path], check=True)
        if result.returncode == 0:
            print("🎉 Fixed experiments completed successfully!")
        else:
            print(f"❌ Fixed experiments failed with error code: {result.returncode}")
    except KeyboardInterrupt:
        print("\n⏸️  Experiments paused by user")
        print("   Progress saved to checkpoint. Run again to resume.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Fixed experiments failed with error code: {e.returncode}")

if __name__ == "__main__":
    main()
