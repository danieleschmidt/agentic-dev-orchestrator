#!/usr/bin/env python3
"""
Simplified Generation 1 Validation Script
Tests core functionality without complex API server tests
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        success = result.returncode == 0
        
        if success:
            print(f"  ✅ {description} passed")
        else:
            print(f"  ❌ {description} failed")
            if result.stderr:
                print(f"     Error: {result.stderr[:200]}")
        
        return success
    except Exception as e:
        print(f"  💥 {description} crashed: {e}")
        return False


def main():
    """Simplified Generation 1 validation"""
    print("🚀 GENERATION 1 VALIDATION - MAKE IT WORK (SIMPLIFIED)")
    print("=" * 60)
    
    results = []
    
    # Test 1: Core modules can be imported
    success = run_command(
        "python3 -c 'import ado; import backlog_manager; import autonomous_executor; print(\"All core modules imported successfully\")'",
        "Core module imports"
    )
    results.append(("Core module imports", success))
    
    # Test 2: CLI help works
    success = run_command(
        "python3 ado.py --help > /dev/null",
        "CLI help command"
    )
    results.append(("CLI help command", success))
    
    # Test 3: Backlog manager works
    success = run_command(
        "python3 -c 'from backlog_manager import BacklogManager; bm = BacklogManager(); print(\"Backlog manager works\")'",
        "Backlog manager instantiation"
    )
    results.append(("Backlog manager", success))
    
    # Test 4: Health monitoring works
    success = run_command(
        "python3 -c 'from src.monitoring.health_monitor import HealthMonitor; hm = HealthMonitor(); print(\"Health monitor works\")'",
        "Health monitor instantiation"
    )
    results.append(("Health monitor", success))
    
    # Test 5: Basic validation works
    success = run_command(
        "python3 -c 'from src.security.input_validator import InputValidator; iv = InputValidator(); print(\"Input validator works\")'",
        "Input validator instantiation"
    )
    results.append(("Input validator", success))
    
    # Test 6: Cache manager works
    success = run_command(
        "python3 -c 'from src.cache.cache_manager import get_cache_manager; cm = get_cache_manager(\"/tmp/test_cache\"); print(\"Cache manager works\")'",
        "Cache manager instantiation"
    )
    results.append(("Cache manager", success))
    
    # Calculate results
    passed = sum(1 for _, success in results if success)
    total = len(results)
    percentage = (passed / total) * 100 if total > 0 else 0
    
    print()
    print("=" * 60)
    print("📊 GENERATION 1 RESULTS (SIMPLIFIED)")
    print(f"Tests Passed: {passed}/{total} ({percentage:.1f}%)")
    
    if percentage >= 80:
        print("✅ GENERATION 1 VALIDATION PASSED")
        success_overall = True
    else:
        print("❌ GENERATION 1 VALIDATION FAILED") 
        success_overall = False
    
    # Save results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "generation": 1,
        "tests": {name: success for name, success in results},
        "overall_success": success_overall,
        "percentage": percentage
    }
    
    os.makedirs("validation_results", exist_ok=True)
    with open("validation_results/generation_1_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    return success_overall


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)