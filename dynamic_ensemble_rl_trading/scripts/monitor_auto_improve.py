"""
Monitor auto-improvement loop progress.
"""

import json
from pathlib import Path
from datetime import datetime
import time

def monitor_improvement():
    """Monitor improvement progress."""
    log_path = Path('results/verification/auto_improve_loop.log')
    history_path = Path('results/verification/improvement_history.json')
    
    print("=" * 80)
    print("Auto-improvement loop monitoring")
    print("=" * 80)
    
    # Check history
    if history_path.exists():
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        if history:
            latest = history[-1]
            print(f"\nLatest iteration: {latest['iteration']}")
            print(f"Current consistency: {latest['consistency']:.1f}%")
            print(f"Improved: {'Yes' if latest.get('improved', False) else 'No improvement'}")
            print(f"\nLatest metrics:")
            for k, v in latest['metrics'].items():
                print(f"  {k}: {v:.4f}")
            
            # Overall progress
            print(f"\nOverall progress:")
            consistencies = [h['consistency'] for h in history]
            best_idx = consistencies.index(max(consistencies))
            print(f"  Best consistency: {max(consistencies):.1f}% (iteration {history[best_idx]['iteration']})")
            print(f"  Average consistency: {sum(consistencies)/len(consistencies):.1f}%")
            print(f"  Total iterations: {len(history)}")
        else:
            print("\nHistory is empty.")
    else:
        print("\nHistory file not found. The loop may not have started yet.")
    
    # Check log file
    if log_path.exists():
        print(f"\nLog file: {log_path}")
        print("Recent log (last 10 lines):")
        print("-" * 80)
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(line.rstrip())
    else:
        print("\nLog file not found.")
    
    print("\n" + "=" * 80)
    print("Monitoring complete")
    print("=" * 80)


if __name__ == '__main__':
    monitor_improvement()
