#!/usr/bin/env python3
"""
Quantum Quality Gates System
Comprehensive quality validation with quantum-inspired quality metrics
"""

import subprocess
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import datetime
import logging
import ast
import re

from quantum_task_planner import QuantumTaskPlanner
from quantum_security_validator import QuantumSecurityValidator
from quantum_performance_optimizer import QuantumPerformanceOptimizer


class QualityGateType(Enum):
    """Types of quality gates"""
    SYNTAX_VALIDATION = "syntax_validation"
    SECURITY_SCANNING = "security_scanning"
    PERFORMANCE_BENCHMARKING = "performance_benchmarking"
    CODE_QUALITY = "code_quality"
    QUANTUM_COHERENCE = "quantum_coherence"
    INTEGRATION_TESTING = "integration_testing"
    DEPENDENCY_VALIDATION = "dependency_validation"
    DOCUMENTATION_CHECK = "documentation_check"


class QualityResult(Enum):
    """Quality gate results"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class QualityGateResult:
    """Result of a quality gate check"""
    gate_type: QualityGateType
    result: QualityResult
    score: float  # 0.0 to 100.0
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    quantum_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class QuantumQualityReport:
    """Comprehensive quality report"""
    timestamp: datetime.datetime
    overall_score: float
    gates_passed: int
    gates_failed: int
    gates_warnings: int
    gate_results: List[QualityGateResult] = field(default_factory=list)
    quantum_coherence_score: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    pass_threshold: float = 80.0


class QuantumQualityGates:
    """Quantum-enhanced quality gates system"""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.logger = self._setup_quality_logging()
        self.quality_config = self._load_quality_config()
        self.gate_implementations = self._initialize_gate_implementations()
        
        # Initialize quantum components
        self.quantum_planner = QuantumTaskPlanner(repo_root)
        self.security_validator = QuantumSecurityValidator(repo_root)
        self.performance_optimizer = QuantumPerformanceOptimizer(repo_root)
        
    def _setup_quality_logging(self) -> logging.Logger:
        """Setup quality-specific logging"""
        logger = logging.getLogger("quantum_quality")
        logger.setLevel(logging.INFO)
        
        # Quality logs directory
        quality_logs_dir = self.repo_root / "logs" / "quality"
        quality_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality log handler
        handler = logging.FileHandler(quality_logs_dir / "quantum_quality_gates.log")
        formatter = logging.Formatter(
            '%(asctime)s - QUALITY - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_quality_config(self) -> Dict:
        """Load quality gates configuration"""
        config_file = self.repo_root / ".quantum_quality.json"
        default_config = {
            "pass_threshold": 80.0,
            "syntax_validation": {"enabled": True, "weight": 15.0},
            "security_scanning": {"enabled": True, "weight": 25.0},
            "performance_benchmarking": {"enabled": True, "weight": 20.0},
            "code_quality": {"enabled": True, "weight": 15.0},
            "quantum_coherence": {"enabled": True, "weight": 10.0},
            "integration_testing": {"enabled": True, "weight": 10.0},
            "dependency_validation": {"enabled": True, "weight": 3.0},
            "documentation_check": {"enabled": True, "weight": 2.0}
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                self.logger.warning(f"Failed to load quality config: {e}")
        
        return default_config
    
    def _initialize_gate_implementations(self) -> Dict[QualityGateType, callable]:
        """Initialize quality gate implementations"""
        return {
            QualityGateType.SYNTAX_VALIDATION: self._run_syntax_validation,
            QualityGateType.SECURITY_SCANNING: self._run_security_scanning,
            QualityGateType.PERFORMANCE_BENCHMARKING: self._run_performance_benchmarking,
            QualityGateType.CODE_QUALITY: self._run_code_quality_check,
            QualityGateType.QUANTUM_COHERENCE: self._run_quantum_coherence_check,
            QualityGateType.INTEGRATION_TESTING: self._run_integration_testing,
            QualityGateType.DEPENDENCY_VALIDATION: self._run_dependency_validation,
            QualityGateType.DOCUMENTATION_CHECK: self._run_documentation_check,
        }
    
    def run_all_quality_gates(self) -> QuantumQualityReport:
        """Run all enabled quality gates"""
        self.logger.info("Starting comprehensive quality gate validation")
        
        start_time = time.time()
        gate_results = []
        
        for gate_type in QualityGateType:
            gate_config = self.quality_config.get(gate_type.value, {})
            
            if not gate_config.get("enabled", True):
                self.logger.info(f"Skipping disabled gate: {gate_type.value}")
                continue
            
            try:
                self.logger.info(f"Running quality gate: {gate_type.value}")
                gate_start = time.time()
                
                gate_impl = self.gate_implementations[gate_type]
                result = gate_impl()
                
                result.execution_time = time.time() - gate_start
                gate_results.append(result)
                
                self.logger.info(f"Gate {gate_type.value} completed: {result.result.value} (Score: {result.score:.1f})")
                
            except Exception as e:
                self.logger.error(f"Gate {gate_type.value} failed with exception: {e}")
                error_result = QualityGateResult(
                    gate_type=gate_type,
                    result=QualityResult.FAILED,
                    score=0.0,
                    message=f"Gate execution failed: {str(e)}",
                    details={"exception": str(e), "traceback": traceback.format_exc()},
                    execution_time=time.time() - gate_start
                )
                gate_results.append(error_result)
        
        # Calculate overall score and generate report
        report = self._generate_quality_report(gate_results)
        report.timestamp = datetime.datetime.now()
        
        total_time = time.time() - start_time
        self.logger.info(f"Quality gate validation completed in {total_time:.2f}s")
        self.logger.info(f"Overall score: {report.overall_score:.1f}/100")
        
        return report
    
    def _run_syntax_validation(self) -> QualityGateResult:
        """Run Python syntax validation"""
        self.logger.info("Running syntax validation")
        
        python_files = list(self.repo_root.glob("**/*.py"))
        if not python_files:
            return QualityGateResult(
                gate_type=QualityGateType.SYNTAX_VALIDATION,
                result=QualityResult.SKIPPED,
                score=100.0,
                message="No Python files found to validate"
            )
        
        syntax_errors = []
        files_checked = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # Parse AST to check syntax
                ast.parse(source_code, filename=str(py_file))
                files_checked += 1
                
            except SyntaxError as e:
                syntax_errors.append({
                    "file": str(py_file),
                    "line": e.lineno,
                    "message": e.msg,
                    "text": e.text
                })
            except Exception as e:
                syntax_errors.append({
                    "file": str(py_file),
                    "error": str(e)
                })
        
        # Calculate score
        if syntax_errors:
            error_rate = len(syntax_errors) / len(python_files)
            score = max(0.0, 100.0 * (1.0 - error_rate))
            result = QualityResult.FAILED
            message = f"Found {len(syntax_errors)} syntax errors in {len(python_files)} files"
        else:
            score = 100.0
            result = QualityResult.PASSED
            message = f"All {files_checked} Python files have valid syntax"
        
        return QualityGateResult(
            gate_type=QualityGateType.SYNTAX_VALIDATION,
            result=result,
            score=score,
            message=message,
            details={
                "files_checked": files_checked,
                "syntax_errors": syntax_errors
            }
        )
    
    def _run_security_scanning(self) -> QualityGateResult:
        """Run quantum security scanning"""
        self.logger.info("Running quantum security scanning")
        
        try:
            # Initialize quantum planner and get tasks
            self.quantum_planner.initialize_quantum_system()
            quantum_tasks = list(self.quantum_planner.quantum_tasks.values())
            
            if not quantum_tasks:
                return QualityGateResult(
                    gate_type=QualityGateType.SECURITY_SCANNING,
                    result=QualityResult.SKIPPED,
                    score=100.0,
                    message="No quantum tasks found for security validation"
                )
            
            # Run security validation
            validation_results = self.security_validator.bulk_validate_tasks(quantum_tasks)
            security_report = self.security_validator.generate_security_report(validation_results)
            
            # Calculate score based on security report
            overall_status = security_report["overall_security_status"]
            status_scores = {
                "ACCEPTABLE": 100.0,
                "MEDIUM_RISK": 75.0,
                "HIGH_RISK": 50.0,
                "CRITICAL": 0.0
            }
            
            score = status_scores.get(overall_status, 50.0)
            
            if score >= 80.0:
                result = QualityResult.PASSED
            elif score >= 60.0:
                result = QualityResult.WARNING
            else:
                result = QualityResult.FAILED
            
            message = f"Security status: {overall_status} ({security_report['total_tasks_validated']} tasks validated)"
            
            return QualityGateResult(
                gate_type=QualityGateType.SECURITY_SCANNING,
                result=result,
                score=score,
                message=message,
                details=security_report,
                quantum_metrics={
                    "average_entropy": security_report["quantum_security_metrics"]["average_entropy"],
                    "low_entropy_tasks": security_report["quantum_security_metrics"]["low_entropy_tasks"]
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_type=QualityGateType.SECURITY_SCANNING,
                result=QualityResult.FAILED,
                score=0.0,
                message=f"Security scanning failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _run_performance_benchmarking(self) -> QualityGateResult:
        """Run performance benchmarking"""
        self.logger.info("Running performance benchmarking")
        
        try:
            # Get performance analytics
            analytics = self.performance_optimizer.get_performance_analytics()
            
            # Calculate performance score based on various metrics
            score_components = []
            
            # Cache utilization score
            cache_stats = analytics.get("cache_statistics", {})
            cache_utilization = cache_stats.get("utilization", 0.0)
            cache_score = min(100.0, cache_utilization * 100)
            score_components.append(cache_score)
            
            # Quantum coherence score
            avg_coherence = analytics.get("average_quantum_coherence", 0.0)
            coherence_score = avg_coherence * 100
            score_components.append(coherence_score)
            
            # Resource pool efficiency
            resource_pool_status = analytics.get("resource_pool_status", {})
            max_workers = resource_pool_status.get("max_workers", 1)
            active_tasks = resource_pool_status.get("active_tasks", 0)
            efficiency_score = min(100.0, (active_tasks / max_workers) * 100) if max_workers > 0 else 100.0
            score_components.append(efficiency_score)
            
            # Overall performance score
            overall_score = sum(score_components) / len(score_components) if score_components else 0.0
            
            if overall_score >= 80.0:
                result = QualityResult.PASSED
            elif overall_score >= 60.0:
                result = QualityResult.WARNING
            else:
                result = QualityResult.FAILED
            
            message = f"Performance score: {overall_score:.1f} (Cache: {cache_score:.1f}, Coherence: {coherence_score:.1f})"
            
            return QualityGateResult(
                gate_type=QualityGateType.PERFORMANCE_BENCHMARKING,
                result=result,
                score=overall_score,
                message=message,
                details=analytics,
                quantum_metrics={
                    "quantum_coherence": avg_coherence,
                    "cache_utilization": cache_utilization,
                    "acceleration_factor": analytics.get("quantum_acceleration_factor", 1.0)
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_type=QualityGateType.PERFORMANCE_BENCHMARKING,
                result=QualityResult.FAILED,
                score=0.0,
                message=f"Performance benchmarking failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _run_code_quality_check(self) -> QualityGateResult:
        """Run code quality checks"""
        self.logger.info("Running code quality checks")
        
        python_files = list(self.repo_root.glob("**/*.py"))
        if not python_files:
            return QualityGateResult(
                gate_type=QualityGateType.CODE_QUALITY,
                result=QualityResult.SKIPPED,
                score=100.0,
                message="No Python files found for quality check"
            )
        
        quality_metrics = {
            "total_files": len(python_files),
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "complex_functions": 0,
            "long_functions": 0,
            "documentation_coverage": 0.0
        }
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # Parse AST for analysis
                tree = ast.parse(source_code, filename=str(py_file))
                lines = source_code.split('\n')
                quality_metrics["total_lines"] += len(lines)
                
                # Analyze AST nodes
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        quality_metrics["total_functions"] += 1
                        
                        # Check function complexity (simple heuristic)
                        complexity = self._calculate_function_complexity(node)
                        if complexity > 10:
                            quality_metrics["complex_functions"] += 1
                        
                        # Check function length
                        if hasattr(node, 'end_lineno') and node.end_lineno:
                            func_length = node.end_lineno - node.lineno
                            if func_length > 50:
                                quality_metrics["long_functions"] += 1
                    
                    elif isinstance(node, ast.ClassDef):
                        quality_metrics["total_classes"] += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze {py_file}: {e}")
        
        # Calculate quality score
        score_components = []
        
        # Complexity score
        if quality_metrics["total_functions"] > 0:
            complexity_ratio = quality_metrics["complex_functions"] / quality_metrics["total_functions"]
            complexity_score = max(0.0, 100.0 * (1.0 - complexity_ratio))
            score_components.append(complexity_score)
        
        # Function length score
        if quality_metrics["total_functions"] > 0:
            length_ratio = quality_metrics["long_functions"] / quality_metrics["total_functions"]
            length_score = max(0.0, 100.0 * (1.0 - length_ratio))
            score_components.append(length_score)
        
        # Default score if no functions found
        if not score_components:
            score_components.append(75.0)
        
        overall_score = sum(score_components) / len(score_components)
        
        if overall_score >= 80.0:
            result = QualityResult.PASSED
        elif overall_score >= 60.0:
            result = QualityResult.WARNING
        else:
            result = QualityResult.FAILED
        
        message = f"Code quality score: {overall_score:.1f} ({quality_metrics['total_functions']} functions, {quality_metrics['complex_functions']} complex)"
        
        recommendations = []
        if quality_metrics["complex_functions"] > 0:
            recommendations.append(f"Consider refactoring {quality_metrics['complex_functions']} complex functions")
        if quality_metrics["long_functions"] > 0:
            recommendations.append(f"Consider breaking down {quality_metrics['long_functions']} long functions")
        
        return QualityGateResult(
            gate_type=QualityGateType.CODE_QUALITY,
            result=result,
            score=overall_score,
            message=message,
            details=quality_metrics,
            recommendations=recommendations
        )
    
    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of function (simplified)"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _run_quantum_coherence_check(self) -> QualityGateResult:
        """Run quantum coherence validation"""
        self.logger.info("Running quantum coherence check")
        
        try:
            # Initialize quantum system and get insights
            self.quantum_planner.initialize_quantum_system()
            insights = self.quantum_planner.get_quantum_insights()
            
            system_coherence = insights.get("system_coherence", 0.0)
            entanglement_clusters = insights.get("entanglement_clusters", [])
            quantum_bottlenecks = insights.get("quantum_bottlenecks", [])
            
            # Calculate coherence score
            coherence_score = system_coherence * 100
            
            # Penalize for bottlenecks
            bottleneck_penalty = len(quantum_bottlenecks) * 10
            final_score = max(0.0, coherence_score - bottleneck_penalty)
            
            if final_score >= 70.0:
                result = QualityResult.PASSED
            elif final_score >= 50.0:
                result = QualityResult.WARNING
            else:
                result = QualityResult.FAILED
            
            message = f"Quantum coherence: {system_coherence:.3f} ({len(entanglement_clusters)} clusters, {len(quantum_bottlenecks)} bottlenecks)"
            
            recommendations = insights.get("optimization_suggestions", [])
            
            return QualityGateResult(
                gate_type=QualityGateType.QUANTUM_COHERENCE,
                result=result,
                score=final_score,
                message=message,
                details=insights,
                recommendations=recommendations,
                quantum_metrics={
                    "system_coherence": system_coherence,
                    "entanglement_clusters": len(entanglement_clusters),
                    "quantum_bottlenecks": len(quantum_bottlenecks)
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_type=QualityGateType.QUANTUM_COHERENCE,
                result=QualityResult.FAILED,
                score=0.0,
                message=f"Quantum coherence check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _run_integration_testing(self) -> QualityGateResult:
        """Run integration testing"""
        self.logger.info("Running integration testing")
        
        try:
            # Test quantum system integration
            integration_results = {
                "planner_init": False,
                "security_validation": False,
                "performance_optimization": False,
                "error_recovery": False
            }
            
            # Test planner initialization
            try:
                self.quantum_planner.initialize_quantum_system()
                integration_results["planner_init"] = True
            except Exception as e:
                self.logger.warning(f"Planner initialization failed: {e}")
            
            # Test security validation
            try:
                if self.quantum_planner.quantum_tasks:
                    task = list(self.quantum_planner.quantum_tasks.values())[0]
                    self.security_validator.validate_quantum_task_security(task)
                    integration_results["security_validation"] = True
            except Exception as e:
                self.logger.warning(f"Security validation failed: {e}")
            
            # Test performance optimization
            try:
                self.performance_optimizer.get_performance_analytics()
                integration_results["performance_optimization"] = True
            except Exception as e:
                self.logger.warning(f"Performance optimization failed: {e}")
            
            # Test error recovery (basic check)
            try:
                from quantum_error_recovery import QuantumErrorRecovery
                recovery = QuantumErrorRecovery(str(self.repo_root))
                recovery.get_error_analytics()
                integration_results["error_recovery"] = True
            except Exception as e:
                self.logger.warning(f"Error recovery check failed: {e}")
            
            # Calculate integration score
            passed_tests = sum(integration_results.values())
            total_tests = len(integration_results)
            integration_score = (passed_tests / total_tests) * 100
            
            if integration_score >= 80.0:
                result = QualityResult.PASSED
            elif integration_score >= 60.0:
                result = QualityResult.WARNING
            else:
                result = QualityResult.FAILED
            
            message = f"Integration tests: {passed_tests}/{total_tests} passed"
            
            return QualityGateResult(
                gate_type=QualityGateType.INTEGRATION_TESTING,
                result=result,
                score=integration_score,
                message=message,
                details=integration_results
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_type=QualityGateType.INTEGRATION_TESTING,
                result=QualityResult.FAILED,
                score=0.0,
                message=f"Integration testing failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _run_dependency_validation(self) -> QualityGateResult:
        """Run dependency validation"""
        self.logger.info("Running dependency validation")
        
        try:
            # Check if required files exist
            required_files = [
                "quantum_task_planner.py",
                "quantum_security_validator.py",
                "quantum_error_recovery.py",
                "quantum_performance_optimizer.py",
                "backlog_manager.py"
            ]
            
            missing_files = []
            for file_name in required_files:
                file_path = self.repo_root / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
            
            # Check Python imports
            import_errors = []
            try:
                import yaml
                import numpy as np
                import psutil
            except ImportError as e:
                import_errors.append(str(e))
            
            # Calculate dependency score
            files_score = ((len(required_files) - len(missing_files)) / len(required_files)) * 70
            imports_score = 30 if not import_errors else 0
            dependency_score = files_score + imports_score
            
            if dependency_score >= 90.0:
                result = QualityResult.PASSED
            elif dependency_score >= 70.0:
                result = QualityResult.WARNING
            else:
                result = QualityResult.FAILED
            
            message = f"Dependencies: {len(required_files) - len(missing_files)}/{len(required_files)} files, {len(import_errors)} import errors"
            
            recommendations = []
            if missing_files:
                recommendations.append(f"Missing required files: {', '.join(missing_files)}")
            if import_errors:
                recommendations.append(f"Import errors: {', '.join(import_errors)}")
            
            return QualityGateResult(
                gate_type=QualityGateType.DEPENDENCY_VALIDATION,
                result=result,
                score=dependency_score,
                message=message,
                details={
                    "required_files": required_files,
                    "missing_files": missing_files,
                    "import_errors": import_errors
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_type=QualityGateType.DEPENDENCY_VALIDATION,
                result=QualityResult.FAILED,
                score=0.0,
                message=f"Dependency validation failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _run_documentation_check(self) -> QualityGateResult:
        """Run documentation completeness check"""
        self.logger.info("Running documentation check")
        
        try:
            # Check for documentation files
            doc_files = {
                "README.md": self.repo_root / "README.md",
                "ARCHITECTURE.md": self.repo_root / "ARCHITECTURE.md",
                "CONTRIBUTING.md": self.repo_root / "CONTRIBUTING.md",
            }
            
            existing_docs = []
            missing_docs = []
            
            for doc_name, doc_path in doc_files.items():
                if doc_path.exists():
                    existing_docs.append(doc_name)
                else:
                    missing_docs.append(doc_name)
            
            # Check Python docstrings
            python_files = list(self.repo_root.glob("*.py"))
            documented_functions = 0
            total_functions = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    
                    tree = ast.parse(source_code, filename=str(py_file))
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                            if ast.get_docstring(node):
                                documented_functions += 1
                
                except Exception as e:
                    self.logger.warning(f"Failed to analyze docstrings in {py_file}: {e}")
            
            # Calculate documentation score
            doc_files_score = (len(existing_docs) / len(doc_files)) * 60
            
            if total_functions > 0:
                docstring_score = (documented_functions / total_functions) * 40
            else:
                docstring_score = 40  # No functions to document
            
            documentation_score = doc_files_score + docstring_score
            
            if documentation_score >= 80.0:
                result = QualityResult.PASSED
            elif documentation_score >= 60.0:
                result = QualityResult.WARNING
            else:
                result = QualityResult.FAILED
            
            message = f"Documentation: {len(existing_docs)}/{len(doc_files)} files, {documented_functions}/{total_functions} functions documented"
            
            recommendations = []
            if missing_docs:
                recommendations.append(f"Missing documentation files: {', '.join(missing_docs)}")
            if total_functions > 0 and documented_functions < total_functions:
                recommendations.append(f"Add docstrings to {total_functions - documented_functions} functions")
            
            return QualityGateResult(
                gate_type=QualityGateType.DOCUMENTATION_CHECK,
                result=result,
                score=documentation_score,
                message=message,
                details={
                    "existing_docs": existing_docs,
                    "missing_docs": missing_docs,
                    "documented_functions": documented_functions,
                    "total_functions": total_functions
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_type=QualityGateType.DOCUMENTATION_CHECK,
                result=QualityResult.FAILED,
                score=0.0,
                message=f"Documentation check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _generate_quality_report(self, gate_results: List[QualityGateResult]) -> QuantumQualityReport:
        """Generate comprehensive quality report"""
        # Calculate weighted overall score
        total_weighted_score = 0.0
        total_weight = 0.0
        
        gates_passed = 0
        gates_failed = 0
        gates_warnings = 0
        
        all_recommendations = []
        quantum_coherence_score = 0.0
        security_score = 0.0
        performance_score = 0.0
        
        for result in gate_results:
            gate_config = self.quality_config.get(result.gate_type.value, {})
            weight = gate_config.get("weight", 10.0)
            
            total_weighted_score += result.score * weight
            total_weight += weight
            
            # Count results
            if result.result == QualityResult.PASSED:
                gates_passed += 1
            elif result.result == QualityResult.FAILED:
                gates_failed += 1
            elif result.result == QualityResult.WARNING:
                gates_warnings += 1
            
            # Collect recommendations
            all_recommendations.extend(result.recommendations)
            
            # Extract specific scores
            if result.gate_type == QualityGateType.QUANTUM_COHERENCE:
                quantum_coherence_score = result.score
            elif result.gate_type == QualityGateType.SECURITY_SCANNING:
                security_score = result.score
            elif result.gate_type == QualityGateType.PERFORMANCE_BENCHMARKING:
                performance_score = result.score
        
        # Calculate overall score
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Generate report-level recommendations
        if overall_score < 60.0:
            all_recommendations.insert(0, "Overall quality score is below acceptable threshold")
        if gates_failed > 0:
            all_recommendations.insert(0, f"{gates_failed} quality gates failed - immediate attention required")
        
        return QuantumQualityReport(
            timestamp=datetime.datetime.now(),
            overall_score=overall_score,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            gates_warnings=gates_warnings,
            gate_results=gate_results,
            quantum_coherence_score=quantum_coherence_score,
            security_score=security_score,
            performance_score=performance_score,
            recommendations=list(set(all_recommendations)),  # Remove duplicates
            pass_threshold=self.quality_config.get("pass_threshold", 80.0)
        )
    
    def save_quality_report(self, report: QuantumQualityReport) -> Path:
        """Save quality report to file"""
        reports_dir = self.repo_root / "quality_reports"
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = report.timestamp.strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f"quality_gates_{timestamp}.json"
        
        # Convert report to JSON-serializable format
        report_data = {
            "timestamp": report.timestamp.isoformat(),
            "overall_score": report.overall_score,
            "gates_passed": report.gates_passed,
            "gates_failed": report.gates_failed,
            "gates_warnings": report.gates_warnings,
            "quantum_coherence_score": report.quantum_coherence_score,
            "security_score": report.security_score,
            "performance_score": report.performance_score,
            "recommendations": report.recommendations,
            "pass_threshold": report.pass_threshold,
            "gate_results": [
                {
                    "gate_type": result.gate_type.value,
                    "result": result.result.value,
                    "score": result.score,
                    "message": result.message,
                    "execution_time": result.execution_time,
                    "recommendations": result.recommendations,
                    "quantum_metrics": result.quantum_metrics,
                    "details": result.details
                }
                for result in report.gate_results
            ]
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Also save as latest
        latest_file = reports_dir / "quality_gates_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Quality report saved to: {report_file}")
        return report_file
    
    def print_quality_summary(self, report: QuantumQualityReport):
        """Print quality report summary to console"""
        print("\n" + "="*80)
        print("🔬 QUANTUM QUALITY GATES REPORT")
        print("="*80)
        print(f"⏰ Timestamp: {report.timestamp}")
        print(f"📊 Overall Score: {report.overall_score:.1f}/100.0")
        print(f"✅ Gates Passed: {report.gates_passed}")
        print(f"⚠️  Gates Warnings: {report.gates_warnings}")
        print(f"❌ Gates Failed: {report.gates_failed}")
        print()
        
        # Status indicator
        if report.overall_score >= report.pass_threshold:
            print("🎉 QUALITY GATES: PASSED")
        else:
            print("🚨 QUALITY GATES: FAILED")
        print()
        
        # Key metrics
        print("🔑 Key Metrics:")
        print(f"   🧠 Quantum Coherence: {report.quantum_coherence_score:.1f}/100")
        print(f"   🛡️  Security Score: {report.security_score:.1f}/100")
        print(f"   ⚡ Performance Score: {report.performance_score:.1f}/100")
        print()
        
        # Gate results
        print("📋 Gate Results:")
        for result in report.gate_results:
            status_icon = {
                QualityResult.PASSED: "✅",
                QualityResult.WARNING: "⚠️",
                QualityResult.FAILED: "❌",
                QualityResult.SKIPPED: "⏭️"
            }.get(result.result, "❓")
            
            print(f"   {status_icon} {result.gate_type.value}: {result.score:.1f}/100 - {result.message}")
        print()
        
        # Recommendations
        if report.recommendations:
            print("💡 Recommendations:")
            for i, rec in enumerate(report.recommendations[:10], 1):  # Show top 10
                print(f"   {i}. {rec}")
            if len(report.recommendations) > 10:
                print(f"   ... and {len(report.recommendations) - 10} more")
        
        print("="*80)
    
    def cleanup(self):
        """Cleanup quality gates resources"""
        try:
            self.performance_optimizer.cleanup()
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")


def main():
    """CLI entry point for quantum quality gates"""
    if len(sys.argv) < 2:
        print("Usage: python quantum_quality_gates.py <command>")
        print("Commands: run, report")
        return
    
    command = sys.argv[1]
    quality_gates = QuantumQualityGates()
    
    try:
        if command == "run":
            print("🔬 Running comprehensive quantum quality gates...")
            report = quality_gates.run_all_quality_gates()
            
            # Print summary
            quality_gates.print_quality_summary(report)
            
            # Save report
            report_file = quality_gates.save_quality_report(report)
            print(f"\n📄 Detailed report saved to: {report_file}")
            
            # Exit with appropriate code
            if report.overall_score >= report.pass_threshold:
                print("\n🎉 All quality gates passed!")
                sys.exit(0)
            else:
                print(f"\n🚨 Quality gates failed (Score: {report.overall_score:.1f} < {report.pass_threshold})")
                sys.exit(1)
                
        elif command == "report":
            # Load and display latest report
            latest_report_file = Path("quality_reports/quality_gates_latest.json")
            if latest_report_file.exists():
                with open(latest_report_file, 'r') as f:
                    report_data = json.load(f)
                print(json.dumps(report_data, indent=2))
            else:
                print("No quality report found. Run 'quantum_quality_gates.py run' first.")
                
        else:
            print(f"Unknown command: {command}")
            
    finally:
        quality_gates.cleanup()


if __name__ == "__main__":
    main()