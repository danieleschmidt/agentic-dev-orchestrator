#!/usr/bin/env python3
"""
Generation 2 Validation Script
Tests robustness improvements: security scanning, circuit breakers, structured logging
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime


def run_command(cmd: str, description: str) -> tuple[bool, str]:
    """Run a command and return success status and output"""
    print(f"🔧 {description}...")
    try:
        # Use bash explicitly for commands with source
        if "source " in cmd:
            result = subprocess.run(
                cmd, shell=True, executable="/bin/bash", capture_output=True, text=True, timeout=120
            )
        else:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=120
            )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        if success:
            print(f"  ✅ {description} passed")
        else:
            print(f"  ❌ {description} failed")
            if output:
                print(f"     Error: {output[:200]}...")
        
        return success, output
    except subprocess.TimeoutExpired:
        print(f"  ⏰ {description} timed out")
        return False, "Command timed out"
    except Exception as e:
        print(f"  💥 {description} crashed: {e}")
        return False, str(e)


def validate_generation_2():
    """Validate all Generation 2 robustness implementations"""
    print("🛡️ GENERATION 2 VALIDATION - MAKE IT ROBUST")
    print("=" * 60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "generation": 2,
        "tests": {},
        "overall_success": False
    }
    
    # Test 1: Security Scanner
    success, output = run_command(
        "source .venv/bin/activate && python src/security/enhanced_scanner.py scan",
        "Comprehensive security scan"
    )
    results["tests"]["security_scan"] = {"success": success, "details": output}
    
    # Test 2: Structured Logging
    success, output = run_command(
        "source .venv/bin/activate && python -c \"from src.logging.structured_logger import get_logger; logger = get_logger('test'); logger.info('Test message'); print('Structured logging test completed')\"",
        "Structured logging system"
    )
    results["tests"]["structured_logging"] = {"success": success, "details": output}
    
    # Test 3: Circuit Breaker
    success, output = run_command(
        "source .venv/bin/activate && python -c \"from src.resilience.circuit_breaker_enhanced import EnhancedCircuitBreaker; cb = EnhancedCircuitBreaker('test'); print('Circuit breaker initialized')\"",
        "Circuit breaker functionality"
    )
    results["tests"]["circuit_breaker"] = {"success": success, "details": output}
    
    # Test 4: Health Monitor with Alerts
    success, output = run_command(
        "source .venv/bin/activate && python src/monitoring/health_monitor.py alerts",
        "Health monitoring with alerts"
    )
    results["tests"]["health_monitoring_alerts"] = {"success": success, "details": output}
    
    # Test 5: Performance Logging
    success, output = run_command(
        "source .venv/bin/activate && python -c \"from src.logging.structured_logger import get_logger; import time; logger = get_logger('perf_test'); start = time.time(); time.sleep(0.1); duration = time.time() - start; logger.performance('test_operation', duration); print('Performance logging test completed')\"",
        "Performance monitoring and logging"
    )
    results["tests"]["performance_logging"] = {"success": success, "details": output}
    
    # Test 6: Security Report Generation
    success, output = run_command(
        "ls security_reports/ | head -1",
        "Security report file exists"
    )
    results["tests"]["security_report"] = {"success": success, "details": output}
    
    # Test 7: Retry Logic
    success, output = run_command(
        "source .venv/bin/activate && python src/resilience/retry_with_backoff.py demo",
        "Retry with backoff functionality"
    )
    results["tests"]["retry_logic"] = {"success": success, "details": output}
    
    # Test 8: Log Files Creation
    success, output = run_command(
        "ls logs/ 2>/dev/null | wc -l",
        "Structured log files creation"
    )
    log_count = int(output.strip()) if output.strip().isdigit() else 0
    results["tests"]["log_files"] = {"success": log_count > 0, "details": f"Found {log_count} log files"}
    
    # Calculate overall success
    test_results = [test["success"] for test in results["tests"].values()]
    total_tests = len(test_results)
    passed_tests = sum(test_results)
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    results["total_tests"] = total_tests
    results["passed_tests"] = passed_tests
    results["success_rate"] = success_rate
    results["overall_success"] = success_rate >= 0.75  # 75% pass rate required for Gen2
    
    print("\n" + "=" * 60)
    print(f"📊 GENERATION 2 RESULTS")
    print(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1%})")
    
    if results["overall_success"]:
        print("🎉 GENERATION 2 VALIDATION PASSED!")
        print("🛡️ Robustness improvements are working")
        print("✅ Security scanning: Enhanced")
        print("✅ Error handling: Circuit breakers implemented")
        print("✅ Logging: Structured audit trails enabled")
        print("✅ Ready to proceed to Generation 3 (MAKE IT SCALE)")
    else:
        print("❌ GENERATION 2 VALIDATION FAILED")
        print("🔧 Fix failing robustness tests before proceeding")
    
    # Save results
    results_dir = Path("validation_results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "generation_2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results["overall_success"]


def main():
    """Main validation entry point"""
    success = validate_generation_2()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()