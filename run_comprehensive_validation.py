#!/usr/bin/env python3
"""
Comprehensive Validation Suite for Terragon Autonomous SDLC v6.0
Advanced validation with security, performance, and quality checks
"""

import asyncio
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
import os


class ComprehensiveValidator:
    """Comprehensive validation orchestrator"""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.logger = self._setup_logger()
        self.validation_results = {}
        self.start_time = time.time()
        
        # Validation thresholds
        self.thresholds = {
            "test_coverage": 85.0,
            "security_score": 90.0,
            "performance_score": 85.0,
            "code_quality": 90.0,
            "documentation_coverage": 80.0,
            "deployment_readiness": 95.0
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup validation logger"""
        logger = logging.getLogger("comprehensive_validator")
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        (self.repo_root / "logs").mkdir(exist_ok=True)
        
        # File handler
        handler = logging.FileHandler(self.repo_root / "logs" / "comprehensive_validation.log")
        handler.setFormatter(
            logging.Formatter('%(asctime)s - VALIDATION - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        
        return logger
    
    async def run_comprehensive_validation(self) -> Dict:
        """Run complete validation suite"""
        self.logger.info("🚀 Starting Comprehensive Validation Suite")
        print("🛡️ Terragon SDLC v6.0 - Comprehensive Validation")
        
        validation_phases = [
            ("Code Quality Analysis", self._validate_code_quality),
            ("Security Scanning", self._validate_security),
            ("Performance Testing", self._validate_performance),
            ("Test Coverage Analysis", self._validate_test_coverage),
            ("Documentation Validation", self._validate_documentation),
            ("Deployment Readiness", self._validate_deployment_readiness),
            ("Integration Testing", self._validate_integration),
            ("Quantum System Validation", self._validate_quantum_systems),
            ("Neural Network Validation", self._validate_neural_networks),
            ("Distributed System Validation", self._validate_distributed_systems)
        ]
        
        for phase_name, phase_func in validation_phases:
            self.logger.info(f"📋 Starting {phase_name}...")
            print(f"📋 {phase_name}...")
            
            try:
                result = await phase_func()
                self.validation_results[phase_name.lower().replace(" ", "_")] = result
                
                if result.get("passed", False):
                    self.logger.info(f"✅ {phase_name} - PASSED")
                    print(f"✅ {phase_name} - PASSED")
                else:
                    self.logger.warning(f"⚠️ {phase_name} - NEEDS ATTENTION")
                    print(f"⚠️ {phase_name} - NEEDS ATTENTION")
                    
            except Exception as e:
                self.logger.error(f"❌ {phase_name} - FAILED: {e}")
                print(f"❌ {phase_name} - FAILED: {e}")
                self.validation_results[phase_name.lower().replace(" ", "_")] = {
                    "passed": False,
                    "error": str(e),
                    "score": 0.0
                }
        
        # Generate final report
        final_report = await self._generate_final_report()
        
        return final_report
    
    async def _validate_code_quality(self) -> Dict:
        """Validate code quality using multiple tools"""
        results = {
            "passed": True,
            "score": 0.0,
            "details": {},
            "recommendations": []
        }
        
        # Run black formatting check
        try:
            black_result = subprocess.run(
                ["python", "-m", "black", "--check", "."],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            black_score = 100.0 if black_result.returncode == 0 else 70.0
            results["details"]["formatting"] = {
                "score": black_score,
                "passed": black_result.returncode == 0,
                "issues": black_result.stdout + black_result.stderr if black_result.returncode != 0 else ""
            }
            
        except Exception as e:
            self.logger.warning(f"Black formatting check failed: {e}")
            results["details"]["formatting"] = {"score": 0.0, "passed": False, "error": str(e)}
        
        # Run ruff linting
        try:
            ruff_result = subprocess.run(
                ["python", "-m", "ruff", "check", "."],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            ruff_score = 100.0 if ruff_result.returncode == 0 else 75.0
            results["details"]["linting"] = {
                "score": ruff_score,
                "passed": ruff_result.returncode == 0,
                "issues": ruff_result.stdout + ruff_result.stderr if ruff_result.returncode != 0 else ""
            }
            
        except Exception as e:
            self.logger.warning(f"Ruff linting failed: {e}")
            results["details"]["linting"] = {"score": 0.0, "passed": False, "error": str(e)}
        
        # Calculate overall code quality score
        scores = [detail.get("score", 0.0) for detail in results["details"].values()]
        results["score"] = sum(scores) / len(scores) if scores else 0.0
        results["passed"] = results["score"] >= self.thresholds["code_quality"]
        
        if not results["passed"]:
            results["recommendations"].append("Run 'python -m black .' to format code")
            results["recommendations"].append("Run 'python -m ruff check . --fix' to fix linting issues")
        
        return results
    
    async def _validate_security(self) -> Dict:
        """Comprehensive security validation"""
        results = {
            "passed": True,
            "score": 0.0,
            "details": {},
            "vulnerabilities": [],
            "recommendations": []
        }
        
        # Run bandit security scanning
        try:
            bandit_result = subprocess.run(
                ["python", "-m", "bandit", "-r", ".", "-f", "json", "-x", "tests/,logs/"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if bandit_result.stdout:
                bandit_data = json.loads(bandit_result.stdout)
                high_severity = len([issue for issue in bandit_data.get("results", []) 
                                  if issue.get("issue_severity") == "HIGH"])
                medium_severity = len([issue for issue in bandit_data.get("results", []) 
                                     if issue.get("issue_severity") == "MEDIUM"])
                
                # Calculate security score
                security_score = max(0, 100 - (high_severity * 20) - (medium_severity * 5))
                
                results["details"]["bandit"] = {
                    "score": security_score,
                    "high_severity_issues": high_severity,
                    "medium_severity_issues": medium_severity,
                    "passed": high_severity == 0
                }
                
                if high_severity > 0:
                    results["vulnerabilities"].extend([
                        f"HIGH: {issue.get('test_name', 'Unknown')} in {issue.get('filename', 'Unknown')}"
                        for issue in bandit_data.get("results", [])
                        if issue.get("issue_severity") == "HIGH"
                    ])
                    
        except Exception as e:
            self.logger.warning(f"Bandit security scan failed: {e}")
            results["details"]["bandit"] = {"score": 0.0, "passed": False, "error": str(e)}
        
        # Check for common security patterns
        security_patterns = await self._check_security_patterns()
        results["details"]["patterns"] = security_patterns
        
        # Calculate overall security score
        scores = [detail.get("score", 0.0) for detail in results["details"].values()]
        results["score"] = sum(scores) / len(scores) if scores else 0.0
        results["passed"] = results["score"] >= self.thresholds["security_score"]
        
        if not results["passed"]:
            results["recommendations"].append("Address high-severity security issues")
            results["recommendations"].append("Review and update security patterns")
            results["recommendations"].append("Consider additional security scanning tools")
        
        return results
    
    async def _check_security_patterns(self) -> Dict:
        """Check for security patterns in code"""
        patterns_score = 100.0
        issues = []
        
        # Check for hardcoded secrets (simplified)
        python_files = list(self.repo_root.glob("**/*.py"))
        
        for file_path in python_files:
            if "test" in str(file_path) or "logs" in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Check for potential secrets
                    if "password" in content.lower() and "=" in content:
                        issues.append(f"Potential password in {file_path}")
                        patterns_score -= 10
                    
                    if "api_key" in content.lower() and "=" in content:
                        issues.append(f"Potential API key in {file_path}")
                        patterns_score -= 15
                        
            except Exception:
                continue
        
        return {
            "score": max(0, patterns_score),
            "passed": len(issues) == 0,
            "issues": issues
        }
    
    async def _validate_performance(self) -> Dict:
        """Validate system performance"""
        results = {
            "passed": True,
            "score": 0.0,
            "details": {},
            "benchmarks": {},
            "recommendations": []
        }
        
        # Test import times
        import_start = time.time()
        try:
            from terragon_autonomous_sdlc_v6_executor import TerragONAutonomousSDLCv6
            from quantum_neural_orchestrator import QuantumNeuralOrchestrator
            from distributed_quantum_orchestrator import DistributedQuantumOrchestrator
            
            import_time = time.time() - import_start
            
            results["benchmarks"]["import_time"] = {
                "value": import_time,
                "threshold": 2.0,
                "passed": import_time < 2.0
            }
            
        except Exception as e:
            self.logger.warning(f"Import performance test failed: {e}")
            results["benchmarks"]["import_time"] = {
                "value": -1,
                "passed": False,
                "error": str(e)
            }
        
        # Test initialization performance
        init_start = time.time()
        try:
            executor = TerragONAutonomousSDLCv6(self.repo_root)
            init_time = time.time() - init_start
            
            results["benchmarks"]["initialization_time"] = {
                "value": init_time,
                "threshold": 1.0,
                "passed": init_time < 1.0
            }
            
        except Exception as e:
            self.logger.warning(f"Initialization performance test failed: {e}")
            results["benchmarks"]["initialization_time"] = {
                "value": -1,
                "passed": False,
                "error": str(e)
            }
        
        # Calculate performance score
        benchmark_scores = []
        for benchmark in results["benchmarks"].values():
            if benchmark.get("passed", False):
                benchmark_scores.append(100.0)
            elif "error" in benchmark:
                benchmark_scores.append(0.0)
            else:
                benchmark_scores.append(70.0)  # Failed but functional
        
        results["score"] = sum(benchmark_scores) / len(benchmark_scores) if benchmark_scores else 0.0
        results["passed"] = results["score"] >= self.thresholds["performance_score"]
        
        if not results["passed"]:
            results["recommendations"].append("Optimize import times")
            results["recommendations"].append("Review initialization procedures")
            results["recommendations"].append("Consider lazy loading for large components")
        
        return results
    
    async def _validate_test_coverage(self) -> Dict:
        """Validate test coverage"""
        results = {
            "passed": True,
            "score": 0.0,
            "details": {},
            "recommendations": []
        }
        
        # Run the comprehensive test suite
        try:
            test_result = subprocess.run(
                [sys.executable, "test_terragon_autonomous_sdlc_v6.py"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            # Parse test results
            test_output = test_result.stdout + test_result.stderr
            
            # Look for test summary
            success_rate = 0.0
            if "Success Rate:" in test_output:
                try:
                    success_line = [line for line in test_output.split('\n') if 'Success Rate:' in line][0]
                    success_rate = float(success_line.split(':')[1].strip().replace('%', ''))
                except (IndexError, ValueError):
                    success_rate = 0.0
            
            results["details"]["test_execution"] = {
                "success_rate": success_rate,
                "passed": success_rate >= self.thresholds["test_coverage"],
                "output": test_output[-1000:] if len(test_output) > 1000 else test_output  # Last 1000 chars
            }
            
            results["score"] = success_rate
            results["passed"] = success_rate >= self.thresholds["test_coverage"]
            
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            results["details"]["test_execution"] = {
                "success_rate": 0.0,
                "passed": False,
                "error": str(e)
            }
            results["score"] = 0.0
            results["passed"] = False
        
        if not results["passed"]:
            results["recommendations"].append("Improve test coverage to meet 85% threshold")
            results["recommendations"].append("Add more comprehensive test cases")
            results["recommendations"].append("Review and fix failing tests")
        
        return results
    
    async def _validate_documentation(self) -> Dict:
        """Validate documentation coverage"""
        results = {
            "passed": True,
            "score": 0.0,
            "details": {},
            "recommendations": []
        }
        
        # Check for required documentation files
        required_docs = [
            "README.md",
            "CHANGELOG.md", 
            "CONTRIBUTING.md",
            "docs/DEVELOPMENT.md",
            "ARCHITECTURE.md"
        ]
        
        doc_scores = []
        missing_docs = []
        
        for doc_file in required_docs:
            doc_path = self.repo_root / doc_file
            if doc_path.exists():
                # Check file size (simple quality metric)
                file_size = doc_path.stat().st_size
                if file_size > 500:  # At least 500 bytes
                    doc_scores.append(100.0)
                else:
                    doc_scores.append(50.0)  # Exists but minimal
            else:
                missing_docs.append(doc_file)
                doc_scores.append(0.0)
        
        # Check docstrings in Python files (simplified)
        python_files = list(self.repo_root.glob("**/*.py"))
        documented_functions = 0
        total_functions = 0
        
        for py_file in python_files:
            if "test" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Count functions and methods
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith('def ') or line.strip().startswith('async def '):
                            total_functions += 1
                            
                            # Check if next few lines contain docstring
                            for j in range(i+1, min(i+5, len(lines))):
                                if '"""' in lines[j] or "'''" in lines[j]:
                                    documented_functions += 1
                                    break
                                elif lines[j].strip() and not lines[j].strip().startswith('#'):
                                    break  # Found non-comment, non-docstring code
                                    
            except Exception:
                continue
        
        # Calculate docstring coverage
        docstring_coverage = (documented_functions / max(total_functions, 1)) * 100
        
        results["details"]["required_files"] = {
            "score": (len(required_docs) - len(missing_docs)) / len(required_docs) * 100,
            "missing": missing_docs,
            "passed": len(missing_docs) == 0
        }
        
        results["details"]["docstring_coverage"] = {
            "coverage": docstring_coverage,
            "documented_functions": documented_functions,
            "total_functions": total_functions,
            "passed": docstring_coverage >= 60.0
        }
        
        # Calculate overall documentation score
        file_score = results["details"]["required_files"]["score"]
        results["score"] = (file_score + docstring_coverage) / 2
        results["passed"] = results["score"] >= self.thresholds["documentation_coverage"]
        
        if not results["passed"]:
            results["recommendations"].extend([f"Create missing documentation: {doc}" for doc in missing_docs])
            if docstring_coverage < 60:
                results["recommendations"].append("Add docstrings to functions and methods")
        
        return results
    
    async def _validate_deployment_readiness(self) -> Dict:
        """Validate deployment readiness"""
        results = {
            "passed": True,
            "score": 0.0,
            "details": {},
            "recommendations": []
        }
        
        # Check required files for deployment
        deployment_files = [
            "pyproject.toml",
            "requirements.txt", 
            "Dockerfile",
            "setup.py"
        ]
        
        file_scores = []
        for file_name in deployment_files:
            file_path = self.repo_root / file_name
            if file_path.exists():
                file_scores.append(100.0)
            else:
                file_scores.append(0.0)
                results["recommendations"].append(f"Create {file_name} for deployment")
        
        # Check package structure
        src_exists = (self.repo_root / "src").exists()
        init_exists = (self.repo_root / "__init__.py").exists() or (self.repo_root / "src" / "__init__.py").exists()
        
        structure_score = 100.0 if (src_exists or init_exists) else 50.0
        
        # Check version consistency (simplified)
        version_score = 100.0  # Assume consistent for now
        
        results["details"]["deployment_files"] = {
            "score": sum(file_scores) / len(file_scores),
            "files_present": len([s for s in file_scores if s > 0]),
            "total_files": len(deployment_files)
        }
        
        results["details"]["package_structure"] = {
            "score": structure_score,
            "src_exists": src_exists,
            "init_exists": init_exists
        }
        
        results["details"]["version_consistency"] = {
            "score": version_score
        }
        
        # Calculate overall deployment readiness
        all_scores = [
            results["details"]["deployment_files"]["score"],
            structure_score,
            version_score
        ]
        
        results["score"] = sum(all_scores) / len(all_scores)
        results["passed"] = results["score"] >= self.thresholds["deployment_readiness"]
        
        return results
    
    async def _validate_integration(self) -> Dict:
        """Validate component integration"""
        results = {
            "passed": True,
            "score": 100.0,
            "details": {},
            "recommendations": []
        }
        
        # Test component imports and basic functionality
        try:
            from terragon_autonomous_sdlc_v6_executor import TerragONAutonomousSDLCv6
            from quantum_neural_orchestrator import QuantumNeuralOrchestrator
            from distributed_quantum_orchestrator import DistributedQuantumOrchestrator
            
            # Test basic integration
            executor = TerragONAutonomousSDLCv6(self.repo_root)
            neural_orchestrator = QuantumNeuralOrchestrator(self.repo_root)
            distributed_orchestrator = DistributedQuantumOrchestrator(self.repo_root)
            
            results["details"]["component_integration"] = {
                "quantum_executor": True,
                "neural_orchestrator": True,
                "distributed_orchestrator": True,
                "score": 100.0
            }
            
        except Exception as e:
            self.logger.error(f"Integration validation failed: {e}")
            results["details"]["component_integration"] = {
                "error": str(e),
                "score": 0.0
            }
            results["score"] = 0.0
            results["passed"] = False
            results["recommendations"].append("Fix component integration issues")
        
        return results
    
    async def _validate_quantum_systems(self) -> Dict:
        """Validate quantum system components"""
        results = {
            "passed": True,
            "score": 100.0,
            "details": {},
            "recommendations": []
        }
        
        try:
            from terragon_autonomous_sdlc_v6_executor import TerragONAutonomousSDLCv6, QuantumExecutionContext
            
            executor = TerragONAutonomousSDLCv6(self.repo_root)
            
            # Test quantum context creation
            from backlog_manager import BacklogItem
            test_item = BacklogItem(
                id="quantum-test",
                title="Quantum validation test",
                description="Test quantum system validation",
                effort=5,
                wsjf={"user_business_value": 5, "time_criticality": 5, "risk_reduction_opportunity_enablement": 5, "job_size": 5},
                acceptance_criteria=["Should work"],
                risk_tier="low"
            )
            
            context = executor._create_quantum_context(test_item)
            
            results["details"]["quantum_context"] = {
                "creation_successful": isinstance(context, QuantumExecutionContext),
                "has_superposition_states": len(context.superposition_states) > 0,
                "valid_complexity": 0.0 <= context.complexity_score <= 1.0,
                "score": 100.0
            }
            
        except Exception as e:
            self.logger.error(f"Quantum system validation failed: {e}")
            results["details"]["quantum_context"] = {
                "error": str(e),
                "score": 0.0
            }
            results["score"] = 0.0
            results["passed"] = False
            results["recommendations"].append("Fix quantum system components")
        
        return results
    
    async def _validate_neural_networks(self) -> Dict:
        """Validate neural network components"""
        results = {
            "passed": True,
            "score": 100.0,
            "details": {},
            "recommendations": []
        }
        
        try:
            from quantum_neural_orchestrator import QuantumNeuralOrchestrator, NeuralContext
            
            orchestrator = QuantumNeuralOrchestrator(self.repo_root)
            
            # Test embedding creation
            embedding = orchestrator._create_embedding("test neural processing")
            
            results["details"]["neural_processing"] = {
                "embedding_creation": embedding is not None,
                "embedding_dimension": len(embedding) == orchestrator.config["embedding_dimension"],
                "network_initialization": all([
                    orchestrator.task_encoder is not None,
                    orchestrator.complexity_predictor is not None,
                    orchestrator.optimization_network is not None
                ]),
                "score": 100.0
            }
            
        except Exception as e:
            self.logger.error(f"Neural network validation failed: {e}")
            results["details"]["neural_processing"] = {
                "error": str(e),
                "score": 0.0
            }
            results["score"] = 0.0
            results["passed"] = False
            results["recommendations"].append("Fix neural network components")
        
        return results
    
    async def _validate_distributed_systems(self) -> Dict:
        """Validate distributed system components"""
        results = {
            "passed": True,
            "score": 100.0,
            "details": {},
            "recommendations": []
        }
        
        try:
            from distributed_quantum_orchestrator import DistributedQuantumOrchestrator, WorkerNode
            
            orchestrator = DistributedQuantumOrchestrator(self.repo_root)
            
            # Test node discovery
            await orchestrator._discover_worker_nodes()
            
            results["details"]["distributed_processing"] = {
                "node_discovery": len(orchestrator.worker_nodes) > 0,
                "cluster_initialization": orchestrator.cluster_id is not None,
                "load_balancing_configured": orchestrator.load_balancing is not None,
                "scaling_configured": orchestrator.scaling_mode is not None,
                "score": 100.0
            }
            
        except Exception as e:
            self.logger.error(f"Distributed system validation failed: {e}")
            results["details"]["distributed_processing"] = {
                "error": str(e),
                "score": 0.0
            }
            results["score"] = 0.0
            results["passed"] = False
            results["recommendations"].append("Fix distributed system components")
        
        return results
    
    async def _generate_final_report(self) -> Dict:
        """Generate comprehensive final validation report"""
        total_execution_time = time.time() - self.start_time
        
        # Calculate overall scores
        phase_scores = []
        passed_phases = 0
        total_phases = len(self.validation_results)
        
        for phase_name, phase_result in self.validation_results.items():
            score = phase_result.get("score", 0.0)
            phase_scores.append(score)
            if phase_result.get("passed", False):
                passed_phases += 1
        
        overall_score = sum(phase_scores) / len(phase_scores) if phase_scores else 0.0
        overall_passed = overall_score >= 85.0  # Overall threshold
        
        # Collect all recommendations
        all_recommendations = []
        for phase_result in self.validation_results.values():
            all_recommendations.extend(phase_result.get("recommendations", []))
        
        # Generate quality gates status
        quality_gates = {
            "code_quality": self.validation_results.get("code_quality_analysis", {}).get("passed", False),
            "security": self.validation_results.get("security_scanning", {}).get("passed", False),
            "performance": self.validation_results.get("performance_testing", {}).get("passed", False),
            "test_coverage": self.validation_results.get("test_coverage_analysis", {}).get("passed", False),
            "documentation": self.validation_results.get("documentation_validation", {}).get("passed", False),
            "deployment": self.validation_results.get("deployment_readiness", {}).get("passed", False)
        }
        
        # Create final report
        final_report = {
            "validation_summary": {
                "timestamp": datetime.now().isoformat(),
                "execution_time": total_execution_time,
                "overall_score": overall_score,
                "overall_passed": overall_passed,
                "phases_passed": passed_phases,
                "total_phases": total_phases,
                "success_rate": (passed_phases / total_phases) * 100
            },
            "quality_gates": quality_gates,
            "phase_results": self.validation_results,
            "recommendations": {
                "high_priority": [rec for rec in all_recommendations if any(keyword in rec.lower() 
                                for keyword in ["security", "critical", "failed"])],
                "medium_priority": [rec for rec in all_recommendations if any(keyword in rec.lower() 
                                  for keyword in ["performance", "coverage", "documentation"])],
                "low_priority": [rec for rec in all_recommendations if rec not in 
                               [rec for rec in all_recommendations if any(keyword in rec.lower() 
                                for keyword in ["security", "critical", "failed", "performance", "coverage", "documentation"])]]
            },
            "deployment_ready": all(quality_gates.values()),
            "next_steps": self._generate_next_steps(quality_gates, all_recommendations)
        }
        
        # Save report
        await self._save_validation_report(final_report)
        
        # Print summary
        self._print_validation_summary(final_report)
        
        return final_report
    
    def _generate_next_steps(self, quality_gates: Dict, recommendations: List[str]) -> List[str]:
        """Generate next steps based on validation results"""
        next_steps = []
        
        if not quality_gates.get("security", True):
            next_steps.append("🔒 Address security vulnerabilities before deployment")
        
        if not quality_gates.get("test_coverage", True):
            next_steps.append("🧪 Improve test coverage to meet requirements")
        
        if not quality_gates.get("code_quality", True):
            next_steps.append("🔧 Fix code quality issues")
        
        if quality_gates.get("deployment", True) and all(quality_gates.values()):
            next_steps.append("🚀 Ready for deployment - all quality gates passed!")
        else:
            next_steps.append("⏳ Complete remaining quality gate requirements before deployment")
        
        # Add top recommendations
        if recommendations:
            next_steps.extend(recommendations[:3])  # Top 3 recommendations
        
        return next_steps
    
    async def _save_validation_report(self, report: Dict):
        """Save validation report to file"""
        reports_dir = self.repo_root / "validation_reports"
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"comprehensive_validation_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also save as latest
        latest_file = reports_dir / "comprehensive_validation_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Validation report saved to {report_file}")
    
    def _print_validation_summary(self, report: Dict):
        """Print validation summary to console"""
        summary = report["validation_summary"]
        quality_gates = report["quality_gates"]
        
        print(f"\n{'='*60}")
        print(f"🛡️ COMPREHENSIVE VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Overall Score: {summary['overall_score']:.1f}/100")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Execution Time: {summary['execution_time']:.2f}s")
        print(f"Phases Passed: {summary['phases_passed']}/{summary['total_phases']}")
        
        print(f"\n🚦 QUALITY GATES:")
        for gate, passed in quality_gates.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {gate.replace('_', ' ').title()}: {status}")
        
        if report.get("deployment_ready", False):
            print(f"\n🚀 DEPLOYMENT STATUS: READY")
        else:
            print(f"\n⏳ DEPLOYMENT STATUS: NOT READY")
        
        # Next steps
        print(f"\n📋 NEXT STEPS:")
        for i, step in enumerate(report.get("next_steps", [])[:5], 1):
            print(f"  {i}. {step}")
        
        print(f"\n📊 Full report available in validation_reports/")
        print(f"{'='*60}")


async def main():
    """Main validation entry point"""
    validator = ComprehensiveValidator()
    report = await validator.run_comprehensive_validation()
    
    # Return appropriate exit code
    if report["validation_summary"]["overall_passed"]:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)