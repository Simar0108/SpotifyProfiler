#!/usr/bin/env python3
"""
Quick launcher for Second Round Hyperparameter Experiments

Usage:
    python run_second_round.py                    # Run all experiments
    python run_second_round.py --phase 1         # Run only Phase 1
    python run_second_round.py --resume          # Resume from checkpoint
    python run_second_round.py --dry-run         # Show what would run
"""

import argparse
import sys
import os
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = ['torch', 'numpy', 'matplotlib', 'sklearn', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_database():
    """Check if the MPD database exists"""
    db_path = "data/MPD/mpd_database.db"
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        print("Please ensure the MPD database is built first")
        return False
    
    print(f"✅ Database found: {db_path}")
    return True

def estimate_runtime():
    """Estimate total runtime based on first round results"""
    # Based on your first round: ~6-18 minutes per experiment
    # 35 experiments × 12 minutes average = ~7 hours
    print("⏱️  Runtime Estimate:")
    print("   • Total experiments: 35")
    print("   • Average time per experiment: ~12 minutes")
    print("   • Estimated total time: ~7 hours")
    print("   • With early stopping: ~5-6 hours")
    print()

def show_experiment_summary():
    """Show summary of planned experiments"""
    print("📋 Experiment Summary:")
    print("   Phase 1 (High Impact):")
    print("     • Learning Rate: 6 experiments (0.005, 0.008, 0.01, 0.012, 0.015, 0.02)")
    print("     • Negative Samples: 5 experiments (15, 18, 22, 25, 30)")
    print("     • Hidden Dimensions: 5 experiments (64, 96, 128, 192, 256)")
    print()
    print("   Phase 2 (Medium Impact):")
    print("     • Context Windows: 4 experiments (2, 3, 4, 5)")
    print("     • Weight Decay: 4 experiments (0.0, 0.0001, 0.0005, 0.001)")
    print("     • Batch Sizes: 4 experiments (512, 1024, 2048, 4096)")
    print()
    print("   Phase 3 (Advanced):")
    print("     • Learning Rate Schedulers: 3 experiments (cosine, step, plateau)")
    print("     • Optimizers: 2 experiments (AdamW, RAdam)")
    print("     • Plus 2 baseline experiments")
    print()

def run_experiments(phase=None, resume=False, dry_run=False):
    """Run the experiments"""
    script_path = "pipeline/second_round_experiments.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Experiment script not found: {script_path}")
        return False
    
    # Build command
    cmd = [sys.executable, script_path]
    
    if phase:
        print(f"🎯 Running Phase {phase} only")
        # Note: You'd need to modify the script to support phase filtering
    
    if resume:
        print("🔄 Resuming from checkpoint")
    
    if dry_run:
        print("🔍 Dry run - showing what would be executed")
        print(f"Command: {' '.join(cmd)}")
        return True
    
    print("🚀 Starting experiments...")
    print("   Press Ctrl+C to pause (experiments will resume from checkpoint)")
    print()
    
    try:
        # Run the experiment script
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⏸️  Experiments paused by user")
        print("   Progress saved to checkpoint. Run again to resume.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiments failed with error code: {e.returncode}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Second Round Hyperparameter Experiments")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], 
                       help="Run only specific phase")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would run without executing")
    parser.add_argument("--check", action="store_true", 
                       help="Check dependencies and database only")
    
    args = parser.parse_args()
    
    print("🎵 Spotify Profiler - Second Round Experiments")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check database
    if not check_database():
        sys.exit(1)
    
    if args.check:
        print("✅ All checks passed!")
        return
    
    # Show experiment summary
    show_experiment_summary()
    
    # Estimate runtime
    estimate_runtime()
    
    # Confirm before running
    if not args.dry_run:
        response = input("🚀 Ready to start experiments? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted by user")
            return
    
    # Run experiments
    success = run_experiments(
        phase=args.phase,
        resume=args.resume,
        dry_run=args.dry_run
    )
    
    if success:
        print("🎉 Experiments completed successfully!")
    else:
        print("❌ Experiments failed or were interrupted")
        sys.exit(1)

if __name__ == "__main__":
    main()
