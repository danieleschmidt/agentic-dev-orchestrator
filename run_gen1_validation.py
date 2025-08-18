#!/usr/bin/env python3
"""
Generation 1 Validation Script
Tests all basic functionality implementations and validates quality gates
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
            print(f"     Error: {output}")
        
        return success, output
    except subprocess.TimeoutExpired:
        print(f"  ⏰ {description} timed out")
        return False, "Command timed out"
    except Exception as e:
        print(f"  💥 {description} crashed: {e}")
        return False, str(e)


def validate_generation_1():
    """Validate all Generation 1 implementations"""
    print("🚀 GENERATION 1 VALIDATION - MAKE IT WORK")
    print("=" * 60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "generation": 1,
        "tests": {},
        "overall_success": False
    }
    
    # Test 1: Unit Tests Pass
    success, output = run_command(
        "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key python3 -m pytest tests/unit/ -v --tb=short",
        "Unit tests execution"
    )
    results["tests"]["unit_tests"] = {"success": success, "details": output}
    
    # Test 2: Core CLI Functionality
    success, output = run_command(
        "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key python3 ado.py --help",
        "CLI help command"
    )
    results["tests"]["cli_help"] = {"success": success, "details": output}
    
    success, output = run_command(
        "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key python3 ado.py status",
        "CLI status command"
    )
    results["tests"]["cli_status"] = {"success": success, "details": output}
    
    # Test 3: API Server Functionality
    # Start API server in background
    run_command("pkill -f 'python src/api/server.py' 2>/dev/null || true", "Clean existing API servers")
    
    server_cmd = "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key ADO_DISABLE_AUTH=true python3 src/api/server.py --host 127.0.0.1 --port 8082"
    server_process = subprocess.Popen(
        server_cmd, shell=True, executable="/bin/bash",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    import time
    time.sleep(5)
    
    # Test API endpoints
    success, output = run_command(
        "curl -s http://127.0.0.1:8082/health",
        "API health endpoint"
    )
    results["tests"]["api_health"] = {"success": success, "details": output}
    
    if success:
        success, output = run_command(
            "curl -s http://127.0.0.1:8082/api/v1/backlog?limit=1",
            "API backlog endpoint"
        )
        results["tests"]["api_backlog"] = {"success": success, "details": output}
    else:
        results["tests"]["api_backlog"] = {"success": False, "details": "Health endpoint failed"}
    
    # Clean up
    server_process.terminate()
    run_command("pkill -f 'python src/api/server.py' 2>/dev/null || true", "Stop test API server")
    
    # Test 4: Health Monitoring
    success, output = run_command(
        "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key python3 src/monitoring/health_monitor.py report",
        "Health monitoring report"
    )
    results["tests"]["health_monitoring"] = {"success": success, "details": output}
    
    # Test 5: Backlog Management
    success, output = run_command(
        "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key python3 backlog_manager.py status",
        "Backlog management"
    )
    results["tests"]["backlog_management"] = {"success": success, "details": output}
    
    # Test 6: Data Layer Functionality
    success, output = run_command(
        "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key python3 -m pytest tests/unit/test_data_layer.py -v",
        "Data layer tests"
    )
    results["tests"]["data_layer"] = {"success": success, "details": output}
    
    # Calculate overall success
    test_results = [test["success"] for test in results["tests"].values()]
    total_tests = len(test_results)
    passed_tests = sum(test_results)
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    results["total_tests"] = total_tests
    results["passed_tests"] = passed_tests
    results["success_rate"] = success_rate
    results["overall_success"] = success_rate >= 0.8  # 80% pass rate required
    
    print("\n" + "=" * 60)
    print(f"📊 GENERATION 1 RESULTS")
    print(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1%})")
    
    if results["overall_success"]:
        print("🎉 GENERATION 1 VALIDATION PASSED!")
        print("✅ Basic functionality implementations are working")
        print("✅ Ready to proceed to Generation 2 (MAKE IT ROBUST)")
    else:
        print("❌ GENERATION 1 VALIDATION FAILED")
        print("🔧 Fix failing tests before proceeding to next generation")
    
    # Save results
    results_dir = Path("validation_results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "generation_1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results["overall_success"]


def main():
    """Main validation entry point"""
    success = validate_generation_1()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()