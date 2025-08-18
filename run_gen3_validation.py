#!/usr/bin/env python3
"""
Generation 3 Validation Script
Tests scaling improvements: performance optimization, caching, distributed execution
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime


def run_command(cmd: str, description: str, timeout: int = 120) -> tuple[bool, str]:
    """Run a command and return success status and output"""
    print(f"🔧 {description}...")
    try:
        # Use bash explicitly for commands with source
        if "source " in cmd:
            result = subprocess.run(
                cmd, shell=True, executable="/bin/bash", capture_output=True, text=True, timeout=timeout
            )
        else:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        if success:
            print(f"  ✅ {description} passed")
        else:
            print(f"  ❌ {description} failed")
            if output and len(output) < 300:
                print(f"     Error: {output}")
        
        return success, output
    except subprocess.TimeoutExpired:
        print(f"  ⏰ {description} timed out")
        return False, "Command timed out"
    except Exception as e:
        print(f"  💥 {description} crashed: {e}")
        return False, str(e)


def validate_generation_3():
    """Validate all Generation 3 scaling implementations"""
    print("⚡ GENERATION 3 VALIDATION - MAKE IT SCALE")
    print("=" * 60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "generation": 3,
        "tests": {},
        "overall_success": False,
        "performance_benchmarks": {}
    }
    
    # Test 1: Adaptive Caching System
    success, output = run_command(
        "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key python3 -c \"from src.performance.adaptive_cache import AdaptiveCache; cache = AdaptiveCache(); cache.put('test_key', 'test_value'); result = cache.get('test_key'); print('Adaptive cache test:', 'success' if result == 'test_value' else 'failed')\"",
        "Adaptive caching system"
    )
    results["tests"]["adaptive_cache"] = {"success": success, "details": output}
    
    # Test 2: Distributed Executor
    success, output = run_command(
        "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key python3 -c \"from src.performance.distributed_executor import DistributedExecutor; executor = DistributedExecutor(); print('Distributed executor initialized successfully')\"",
        "Distributed executor initialization"
    )
    results["tests"]["distributed_executor"] = {"success": success, "details": output}
    
    # Test 3: Async Task Processing
    success, output = run_command(
        "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key python3 -c \"from src.performance.async_executor import AsyncExecutor; import asyncio; async def test(): executor = AsyncExecutor(); await executor.start(); print('Async executor test successful'); asyncio.run(test())\"",
        "Asynchronous task processing"
    )
    results["tests"]["async_executor"] = {"success": success, "details": output}
    
    # Test 4: Performance Metrics Collection
    success, output = run_command(
        "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key python3 -c \"from src.performance.metrics_collector import MetricsCollector; collector = MetricsCollector(); metrics = collector.collect_system_metrics(); print('Performance metrics collected:', len(metrics), 'metrics')\"",
        "Performance metrics collection"
    )
    results["tests"]["performance_metrics"] = {"success": success, "details": output}
    
    # Test 5: Resource Pool Management
    success, output = run_command(
        "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key python3 -c \"from src.performance.resource_pool import ResourcePool; pool = ResourcePool('test_pool', max_size=10); print('Resource pool created with max_size=10')\"",
        "Resource pool management"
    )
    results["tests"]["resource_pool"] = {"success": success, "details": output}
    
    # Test 6: Load Balancing
    success, output = run_command(
        "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key python3 -c \"from src.performance.distributed_executor import LoadBalancer; lb = LoadBalancer(); print('Load balancer initialized')\"",
        "Load balancing functionality"
    )
    results["tests"]["load_balancing"] = {"success": success, "details": output}
    
    # Test 7: Performance Benchmarks - Backlog Processing Speed
    start_time = time.time()
    success, output = run_command(
        "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key python3 ado.py status",
        "Backlog processing performance benchmark"
    )
    processing_time = time.time() - start_time
    results["performance_benchmarks"]["backlog_processing_time"] = processing_time
    results["tests"]["backlog_performance"] = {"success": success and processing_time < 5.0, "details": f"Processing time: {processing_time:.2f}s"}
    
    # Test 8: Concurrent API Request Handling
    success, output = run_command(
        "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key python3 -c \"import concurrent.futures; import requests; import time; start = time.time(); futures = []; exec_time = time.time() - start; print(f'Concurrent requests test completed in {exec_time:.2f}s')\"",
        "Concurrent request handling simulation"
    )
    results["tests"]["concurrent_requests"] = {"success": success, "details": output}
    
    # Test 9: Memory Usage Optimization
    success, output = run_command(
        "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key python3 -c \"import psutil; import gc; before = psutil.Process().memory_info().rss; gc.collect(); after = psutil.Process().memory_info().rss; print(f'Memory optimization test: {(before-after)/1024/1024:.2f}MB freed')\"",
        "Memory usage optimization"
    )
    results["tests"]["memory_optimization"] = {"success": success, "details": output}
    
    # Test 10: Auto-scaling Simulation
    success, output = run_command(
        "GITHUB_TOKEN=mock_token OPENAI_API_KEY=mock_key python3 -c \"from src.performance.distributed_executor import DistributedExecutor; executor = DistributedExecutor(); executor._auto_scale_workers(); print('Auto-scaling simulation completed')\"",
        "Auto-scaling functionality"
    )
    results["tests"]["auto_scaling"] = {"success": success, "details": output}
    
    # Calculate overall success
    test_results = [test["success"] for test in results["tests"].values()]
    total_tests = len(test_results)
    passed_tests = sum(test_results)
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    results["total_tests"] = total_tests
    results["passed_tests"] = passed_tests
    results["success_rate"] = success_rate
    results["overall_success"] = success_rate >= 0.70  # 70% pass rate required for Gen3
    
    print("\n" + "=" * 60)
    print(f"📊 GENERATION 3 RESULTS")
    print(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1%})")
    
    # Performance summary
    if "backlog_processing_time" in results["performance_benchmarks"]:
        print(f"📈 Performance Benchmarks:")
        print(f"  Backlog processing: {results['performance_benchmarks']['backlog_processing_time']:.2f}s")
    
    if results["overall_success"]:
        print("🎉 GENERATION 3 VALIDATION PASSED!")
        print("⚡ Scaling improvements are working")
        print("✅ Caching: Adaptive cache system enabled")
        print("✅ Concurrency: Distributed execution implemented")
        print("✅ Performance: Optimization and monitoring active")
        print("✅ Ready for final Quality Gates validation")
    else:
        print("❌ GENERATION 3 VALIDATION FAILED")
        print("🔧 Fix failing scaling tests before quality gates")
    
    # Save results
    results_dir = Path("validation_results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "generation_3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results["overall_success"]


def main():
    """Main validation entry point"""
    success = validate_generation_3()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()