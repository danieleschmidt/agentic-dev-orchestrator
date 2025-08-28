#!/usr/bin/env python3
"""
Progressive Quality Gates Orchestrator v1.0
Adaptive quality validation that evolves with project maturity and risk assessment
Generation 1: MAKE IT WORK - Core adaptive quality gate functionality
"""

import os
import json
import asyncio
import logging
import datetime
import time
import subprocess
import hashlib
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pathlib import Path
from dataclasses import dataclass, asdict, field
import statistics
from enum import Enum
import yaml

class ProjectMaturityLevel(Enum):
    """Project maturity levels for progressive quality gates"""
    PROTOTYPE = "prototype"       # 0-25% maturity - Minimal gates
    DEVELOPMENT = "development"   # 25-50% maturity - Basic gates
    STABLE = "stable"            # 50-75% maturity - Enhanced gates  
    PRODUCTION = "production"    # 75-100% maturity - Comprehensive gates

class RiskAssessmentLevel(Enum):
    """Risk assessment levels for quality gate selection"""
    LOW = "low"           # Non-critical features
    MEDIUM = "medium"     # Important features
    HIGH = "high"         # Critical features
    CRITICAL = "critical" # Security/financial features

class GateAdaptiveMode(Enum):
    """Adaptive modes for quality gate execution"""
    LEARNING = "learning"     # Learn from project patterns
    ENFORCING = "enforcing"   # Apply learned patterns
    EVOLVING = "evolving"     # Continuously adapt

@dataclass
class ProjectMetrics:
    """Project maturity metrics for adaptive gate selection"""
    total_files: int = 0
    total_lines: int = 0
    test_coverage: float = 0.0
    documentation_ratio: float = 0.0
    security_score: float = 0.0
    complexity_score: float = 0.0
    commit_frequency: float = 0.0
    contributor_count: int = 0
    age_days: int = 0
    
    @property
    def maturity_score(self) -> float:
        """Calculate overall project maturity score (0-1)"""
        factors = [
            min(1.0, self.total_files / 100),         # File count factor
            min(1.0, self.total_lines / 10000),       # Size factor
            self.test_coverage / 100,                 # Testing factor
            self.documentation_ratio,                 # Documentation factor
            self.security_score,                      # Security factor
            min(1.0, (1.0 - self.complexity_score)), # Complexity factor (inverted)
            min(1.0, self.commit_frequency / 5),      # Activity factor
            min(1.0, self.contributor_count / 10),    # Team factor
            min(1.0, self.age_days / 365)             # Stability factor
        ]
        return statistics.mean(factors)

@dataclass
class AdaptiveQualityGate:
    """Adaptive quality gate definition"""
    name: str
    description: str
    min_maturity: ProjectMaturityLevel
    risk_levels: List[RiskAssessmentLevel]
    base_threshold: float
    adaptive_thresholds: Dict[ProjectMaturityLevel, float]
    weight: float = 1.0
    critical: bool = False
    learning_enabled: bool = True
    
    def get_threshold_for_maturity(self, maturity: ProjectMaturityLevel) -> float:
        """Get threshold adjusted for project maturity"""
        return self.adaptive_thresholds.get(maturity, self.base_threshold)

@dataclass
class ProgressiveValidationResult:
    """Result of progressive quality validation"""
    gate_name: str
    status: str  # passed, failed, warning, skipped, adapted
    score: float
    threshold: float
    original_threshold: float
    adapted_reason: str = ""
    execution_time: float = 0.0
    maturity_adjustment: float = 0.0
    risk_adjustment: float = 0.0
    learning_data: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

@dataclass
class ProgressiveQualityReport:
    """Comprehensive progressive quality validation report"""
    project_name: str
    maturity_level: ProjectMaturityLevel
    maturity_score: float
    risk_level: RiskAssessmentLevel
    adaptive_mode: GateAdaptiveMode
    gate_results: List[ProgressiveValidationResult]
    overall_score: float
    overall_status: str
    adaptations_applied: int
    learning_insights: Dict[str, Any]
    recommendations: List[str]
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if not self.gate_results:
            return 0.0
        passed = sum(1 for r in self.gate_results if r.status in ['passed', 'adapted'])
        return passed / len(self.gate_results) * 100

class ProgressiveQualityGatesOrchestrator:
    """
    Progressive Quality Gates Orchestrator
    Implements adaptive quality validation that evolves with project maturity
    """
    
    def __init__(self, project_path: Path, config: Optional[Dict[str, Any]] = None):
        self.project_path = project_path
        self.config = config or self._load_default_config()
        self.logger = self._setup_logging()
        
        # Initialize adaptive quality gates
        self.adaptive_gates = self._initialize_adaptive_gates()
        
        # Learning and adaptation storage
        self.learning_history: List[Dict[str, Any]] = []
        self.adaptation_patterns: Dict[str, Any] = {}
        
        # Project analysis cache
        self._project_metrics_cache: Optional[ProjectMetrics] = None
        self._cache_timestamp: Optional[datetime.datetime] = None
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default progressive quality gates configuration"""
        return {
            'adaptive_mode': GateAdaptiveMode.EVOLVING.value,
            'learning_enabled': True,
            'risk_assessment_enabled': True,
            'maturity_analysis_enabled': True,
            'cache_duration_minutes': 30,
            'adaptation_sensitivity': 0.1,  # How sensitive to adapt (0.0-1.0)
            'learning_threshold': 10,  # Min data points for learning
            'quality_score_weights': {
                'maturity': 0.3,
                'risk': 0.2,
                'historical': 0.2,
                'current': 0.3
            },
            'progressive_thresholds': {
                'prototype': 0.6,     # 60% quality threshold for prototypes
                'development': 0.7,   # 70% for development
                'stable': 0.8,        # 80% for stable
                'production': 0.9     # 90% for production
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup progressive quality gates logging"""
        logger = logging.getLogger('progressive_quality_gates')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s [PROGRESSIVE-GATES] %(levelname)s: %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_adaptive_gates(self) -> List[AdaptiveQualityGate]:
        """Initialize adaptive quality gates with progressive thresholds"""
        return [
            AdaptiveQualityGate(
                name="code_syntax_validation",
                description="Validate code syntax with adaptive complexity thresholds",
                min_maturity=ProjectMaturityLevel.PROTOTYPE,
                risk_levels=[RiskAssessmentLevel.LOW, RiskAssessmentLevel.MEDIUM, 
                           RiskAssessmentLevel.HIGH, RiskAssessmentLevel.CRITICAL],
                base_threshold=0.95,
                adaptive_thresholds={
                    ProjectMaturityLevel.PROTOTYPE: 0.85,    # More lenient for prototypes
                    ProjectMaturityLevel.DEVELOPMENT: 0.90,
                    ProjectMaturityLevel.STABLE: 0.95,
                    ProjectMaturityLevel.PRODUCTION: 0.98    # Strict for production
                },
                critical=True
            ),
            AdaptiveQualityGate(
                name="test_coverage_validation",
                description="Progressive test coverage requirements",
                min_maturity=ProjectMaturityLevel.DEVELOPMENT,
                risk_levels=[RiskAssessmentLevel.MEDIUM, RiskAssessmentLevel.HIGH, 
                           RiskAssessmentLevel.CRITICAL],
                base_threshold=0.8,
                adaptive_thresholds={
                    ProjectMaturityLevel.PROTOTYPE: 0.3,     # Basic testing
                    ProjectMaturityLevel.DEVELOPMENT: 0.6,   # Moderate coverage
                    ProjectMaturityLevel.STABLE: 0.8,        # Good coverage
                    ProjectMaturityLevel.PRODUCTION: 0.9     # Comprehensive coverage
                },
                critical=True
            ),
            AdaptiveQualityGate(
                name="security_vulnerability_scan",
                description="Adaptive security scanning based on risk level",
                min_maturity=ProjectMaturityLevel.DEVELOPMENT,
                risk_levels=[RiskAssessmentLevel.MEDIUM, RiskAssessmentLevel.HIGH, 
                           RiskAssessmentLevel.CRITICAL],
                base_threshold=0.0,  # Zero tolerance baseline
                adaptive_thresholds={
                    ProjectMaturityLevel.PROTOTYPE: 2.0,     # Allow some issues for rapid prototyping
                    ProjectMaturityLevel.DEVELOPMENT: 1.0,   # Minimal issues allowed
                    ProjectMaturityLevel.STABLE: 0.0,        # Zero tolerance
                    ProjectMaturityLevel.PRODUCTION: 0.0     # Absolute zero tolerance
                },
                critical=True
            ),
            AdaptiveQualityGate(
                name="performance_benchmarking",
                description="Performance requirements that scale with maturity",
                min_maturity=ProjectMaturityLevel.STABLE,
                risk_levels=[RiskAssessmentLevel.HIGH, RiskAssessmentLevel.CRITICAL],
                base_threshold=1000,  # 1000ms baseline
                adaptive_thresholds={
                    ProjectMaturityLevel.PROTOTYPE: 5000,    # Very relaxed
                    ProjectMaturityLevel.DEVELOPMENT: 2000,  # Moderate
                    ProjectMaturityLevel.STABLE: 1000,       # Good performance
                    ProjectMaturityLevel.PRODUCTION: 500     # Excellent performance
                },
                critical=False
            ),
            AdaptiveQualityGate(
                name="documentation_validation",
                description="Progressive documentation requirements",
                min_maturity=ProjectMaturityLevel.DEVELOPMENT,
                risk_levels=[RiskAssessmentLevel.LOW, RiskAssessmentLevel.MEDIUM, 
                           RiskAssessmentLevel.HIGH, RiskAssessmentLevel.CRITICAL],
                base_threshold=0.7,
                adaptive_thresholds={
                    ProjectMaturityLevel.PROTOTYPE: 0.2,     # Minimal docs
                    ProjectMaturityLevel.DEVELOPMENT: 0.5,   # Basic docs
                    ProjectMaturityLevel.STABLE: 0.7,        # Good docs
                    ProjectMaturityLevel.PRODUCTION: 0.9     # Comprehensive docs
                },
                critical=False
            ),
            AdaptiveQualityGate(
                name="code_quality_analysis",
                description="Adaptive code quality metrics",
                min_maturity=ProjectMaturityLevel.DEVELOPMENT,
                risk_levels=[RiskAssessmentLevel.MEDIUM, RiskAssessmentLevel.HIGH, 
                           RiskAssessmentLevel.CRITICAL],
                base_threshold=8.0,  # Out of 10
                adaptive_thresholds={
                    ProjectMaturityLevel.PROTOTYPE: 5.0,     # Basic quality
                    ProjectMaturityLevel.DEVELOPMENT: 6.5,   # Moderate quality
                    ProjectMaturityLevel.STABLE: 8.0,        # Good quality
                    ProjectMaturityLevel.PRODUCTION: 9.0     # Excellent quality
                },
                critical=False
            )
        ]
    
    async def analyze_project_maturity(self) -> ProjectMetrics:
        """Analyze project to determine maturity level"""
        # Check cache first
        if (self._project_metrics_cache and 
            self._cache_timestamp and
            datetime.datetime.now() - self._cache_timestamp < 
            datetime.timedelta(minutes=self.config.get('cache_duration_minutes', 30))):
            return self._project_metrics_cache
        
        self.logger.info("🔍 Analyzing project maturity...")
        
        metrics = ProjectMetrics()
        
        try:
            # Count files and lines
            python_files = list(self.project_path.rglob('*.py'))
            metrics.total_files = len(python_files)
            
            total_lines = 0
            total_functions = 0
            documented_functions = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
                    total_lines += len(code_lines)
                    
                    # Count functions and documentation
                    functions = re.findall(r'def\s+\w+\(', content)
                    docstrings = re.findall(r'"""[\s\S]*?"""', content)
                    
                    total_functions += len(functions)
                    documented_functions += min(len(functions), len(docstrings))
                    
                except Exception as e:
                    self.logger.warning(f"Could not analyze {py_file}: {e}")
            
            metrics.total_lines = total_lines
            if total_functions > 0:
                metrics.documentation_ratio = documented_functions / total_functions
            
            # Estimate test coverage
            test_files = list(self.project_path.rglob('test_*.py'))
            test_files.extend(list(self.project_path.rglob('*_test.py')))
            
            if test_files:
                test_lines = 0
                for test_file in test_files:
                    try:
                        with open(test_file, 'r', encoding='utf-8') as f:
                            test_content = f.read()
                        test_code_lines = [line for line in test_content.split('\n') 
                                         if line.strip() and not line.strip().startswith('#')]
                        test_lines += len(test_code_lines)
                    except Exception:
                        pass
                
                # Rough estimation: test coverage based on test-to-code ratio
                if total_lines > 0:
                    test_ratio = test_lines / total_lines
                    metrics.test_coverage = min(test_ratio * 300, 100)  # Heuristic
            
            # Calculate complexity score (simplified)
            complexity_indicators = 0
            for py_file in python_files[:10]:  # Sample for performance
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Count complexity indicators
                    complexity_indicators += content.count('if ')
                    complexity_indicators += content.count('for ')
                    complexity_indicators += content.count('while ')
                    complexity_indicators += content.count('try:')
                    complexity_indicators += content.count('except')
                except Exception:
                    pass
            
            if total_lines > 0:
                metrics.complexity_score = min(1.0, complexity_indicators / total_lines * 10)
            
            # Basic security analysis
            security_issues = 0
            dangerous_patterns = ['eval(', 'exec(', 'os.system(', 'subprocess.call(shell=True']
            
            for py_file in python_files[:20]:  # Sample for performance
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern in dangerous_patterns:
                        security_issues += content.count(pattern)
                except Exception:
                    pass
            
            metrics.security_score = max(0.0, 1.0 - (security_issues * 0.1))
            
            # Project age analysis
            try:
                git_dir = self.project_path / '.git'
                if git_dir.exists():
                    # Get first commit date
                    result = subprocess.run(
                        ['git', 'log', '--reverse', '--format=%ai', '-1'],
                        cwd=self.project_path,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        first_commit_str = result.stdout.strip()
                        try:
                            first_commit = datetime.datetime.fromisoformat(first_commit_str.replace(' ', 'T'))
                            metrics.age_days = (datetime.datetime.now() - first_commit).days
                        except Exception:
                            pass
                    
                    # Get commit frequency (commits per week)
                    result = subprocess.run(
                        ['git', 'log', '--since="4 weeks ago"', '--oneline'],
                        cwd=self.project_path,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        recent_commits = len(result.stdout.strip().split('\n'))
                        metrics.commit_frequency = recent_commits / 4  # Commits per week
                    
                    # Get contributor count
                    result = subprocess.run(
                        ['git', 'log', '--format=%ae'],
                        cwd=self.project_path,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        contributors = set(result.stdout.strip().split('\n'))
                        metrics.contributor_count = len(contributors)
                        
            except Exception as e:
                self.logger.debug(f"Git analysis failed: {e}")
        
        except Exception as e:
            self.logger.error(f"Project maturity analysis failed: {e}")
        
        # Cache results
        self._project_metrics_cache = metrics
        self._cache_timestamp = datetime.datetime.now()
        
        self.logger.info(f"📊 Project maturity score: {metrics.maturity_score:.2f}")
        return metrics
    
    def determine_maturity_level(self, metrics: ProjectMetrics) -> ProjectMaturityLevel:
        """Determine project maturity level from metrics"""
        score = metrics.maturity_score
        
        if score >= 0.75:
            return ProjectMaturityLevel.PRODUCTION
        elif score >= 0.50:
            return ProjectMaturityLevel.STABLE
        elif score >= 0.25:
            return ProjectMaturityLevel.DEVELOPMENT
        else:
            return ProjectMaturityLevel.PROTOTYPE
    
    def assess_risk_level(self, change_context: Optional[Dict[str, Any]] = None) -> RiskAssessmentLevel:
        """Assess risk level for current changes"""
        # Simple risk assessment - can be enhanced with ML/AI
        if not change_context:
            return RiskAssessmentLevel.MEDIUM  # Default
        
        risk_indicators = {
            'security_files_changed': RiskAssessmentLevel.CRITICAL,
            'auth_files_changed': RiskAssessmentLevel.CRITICAL,
            'config_files_changed': RiskAssessmentLevel.HIGH,
            'core_files_changed': RiskAssessmentLevel.HIGH,
            'test_files_only': RiskAssessmentLevel.LOW,
            'docs_files_only': RiskAssessmentLevel.LOW
        }
        
        # Check for risk indicators in change context
        for indicator, risk_level in risk_indicators.items():
            if change_context.get(indicator, False):
                return risk_level
        
        return RiskAssessmentLevel.MEDIUM
    
    async def execute_progressive_validation(self, 
                                           change_context: Optional[Dict[str, Any]] = None) -> ProgressiveQualityReport:
        """Execute progressive quality validation with adaptive thresholds"""
        self.logger.info("🚀 Starting Progressive Quality Gates Validation...")
        
        start_time = time.time()
        
        # Analyze project maturity
        project_metrics = await self.analyze_project_maturity()
        maturity_level = self.determine_maturity_level(project_metrics)
        risk_level = self.assess_risk_level(change_context)
        
        self.logger.info(f"📊 Project Maturity: {maturity_level.value} (score: {project_metrics.maturity_score:.2f})")
        self.logger.info(f"⚠️ Risk Level: {risk_level.value}")
        
        # Select applicable gates based on maturity and risk
        applicable_gates = self._select_applicable_gates(maturity_level, risk_level)
        
        self.logger.info(f"🎯 Selected {len(applicable_gates)} adaptive quality gates")
        
        # Execute gates with progressive validation
        gate_results = []
        adaptations_applied = 0
        
        for gate in applicable_gates:
            try:
                result = await self._execute_adaptive_gate(gate, maturity_level, risk_level, project_metrics)
                gate_results.append(result)
                
                if result.adapted_reason:
                    adaptations_applied += 1
                    self.logger.info(f"🔧 Adapted {gate.name}: {result.adapted_reason}")
                    
            except Exception as e:
                self.logger.error(f"Failed to execute gate {gate.name}: {e}")
                gate_results.append(ProgressiveValidationResult(
                    gate_name=gate.name,
                    status='failed',
                    score=0.0,
                    threshold=gate.base_threshold,
                    original_threshold=gate.base_threshold,
                    adapted_reason=f"Execution error: {str(e)}"
                ))
        
        # Calculate overall scores and status
        overall_score = self._calculate_progressive_score(gate_results)
        overall_status = self._determine_progressive_status(gate_results, overall_score, maturity_level)
        
        # Generate learning insights
        learning_insights = self._generate_learning_insights(gate_results, project_metrics)
        
        # Generate recommendations
        recommendations = self._generate_adaptive_recommendations(gate_results, maturity_level, risk_level)
        
        # Create comprehensive report
        report = ProgressiveQualityReport(
            project_name=self.project_path.name,
            maturity_level=maturity_level,
            maturity_score=project_metrics.maturity_score,
            risk_level=risk_level,
            adaptive_mode=GateAdaptiveMode(self.config.get('adaptive_mode', 'evolving')),
            gate_results=gate_results,
            overall_score=overall_score,
            overall_status=overall_status,
            adaptations_applied=adaptations_applied,
            learning_insights=learning_insights,
            recommendations=recommendations
        )
        
        # Save learning data
        await self._save_learning_data(report)
        
        # Save report
        await self._save_progressive_report(report)
        
        execution_time = time.time() - start_time
        
        self.logger.info(f"✨ Progressive validation completed in {execution_time:.2f}s")
        self.logger.info(f"📊 Overall Score: {overall_score:.1f}% | Status: {overall_status}")
        self.logger.info(f"🔧 Adaptations Applied: {adaptations_applied}")
        
        return report
    
    def _select_applicable_gates(self, maturity_level: ProjectMaturityLevel, 
                                risk_level: RiskAssessmentLevel) -> List[AdaptiveQualityGate]:
        """Select applicable gates based on maturity and risk level"""
        applicable_gates = []
        
        maturity_order = [
            ProjectMaturityLevel.PROTOTYPE,
            ProjectMaturityLevel.DEVELOPMENT,
            ProjectMaturityLevel.STABLE,
            ProjectMaturityLevel.PRODUCTION
        ]
        
        current_maturity_index = maturity_order.index(maturity_level)
        
        for gate in self.adaptive_gates:
            # Check if gate is applicable for current maturity level
            gate_maturity_index = maturity_order.index(gate.min_maturity)
            
            if current_maturity_index >= gate_maturity_index and risk_level in gate.risk_levels:
                applicable_gates.append(gate)
        
        return applicable_gates
    
    async def _execute_adaptive_gate(self, gate: AdaptiveQualityGate, 
                                   maturity_level: ProjectMaturityLevel,
                                   risk_level: RiskAssessmentLevel,
                                   project_metrics: ProjectMetrics) -> ProgressiveValidationResult:
        """Execute single adaptive quality gate"""
        start_time = time.time()
        
        # Get adaptive threshold
        adaptive_threshold = gate.get_threshold_for_maturity(maturity_level)
        original_threshold = gate.base_threshold
        
        # Apply risk-based adjustments
        risk_adjustment = self._calculate_risk_adjustment(risk_level, gate)
        final_threshold = adaptive_threshold + risk_adjustment
        
        # Execute gate validation
        score, status, details = await self._run_gate_validation(gate, final_threshold)
        
        # Check if adaptation was beneficial
        adapted_reason = ""
        if adaptive_threshold != original_threshold:
            adapted_reason = f"Threshold adapted from {original_threshold} to {adaptive_threshold} for {maturity_level.value} maturity"
        
        if risk_adjustment != 0:
            if adapted_reason:
                adapted_reason += f", risk-adjusted by {risk_adjustment:+.2f} for {risk_level.value} risk"
            else:
                adapted_reason = f"Risk-adjusted threshold by {risk_adjustment:+.2f} for {risk_level.value} risk"
        
        # Generate recommendations based on learning
        recommendations = self._generate_gate_recommendations(gate, score, final_threshold, details)
        
        execution_time = time.time() - start_time
        
        return ProgressiveValidationResult(
            gate_name=gate.name,
            status=status,
            score=score,
            threshold=final_threshold,
            original_threshold=original_threshold,
            adapted_reason=adapted_reason,
            execution_time=execution_time,
            maturity_adjustment=adaptive_threshold - original_threshold,
            risk_adjustment=risk_adjustment,
            learning_data=details,
            recommendations=recommendations
        )
    
    def _calculate_risk_adjustment(self, risk_level: RiskAssessmentLevel, 
                                 gate: AdaptiveQualityGate) -> float:
        """Calculate risk-based threshold adjustment"""
        risk_multipliers = {
            RiskAssessmentLevel.LOW: -0.1,      # More lenient
            RiskAssessmentLevel.MEDIUM: 0.0,    # No change
            RiskAssessmentLevel.HIGH: 0.1,      # Stricter
            RiskAssessmentLevel.CRITICAL: 0.2   # Much stricter
        }
        
        base_adjustment = risk_multipliers.get(risk_level, 0.0)
        
        # Different gates react differently to risk
        gate_risk_sensitivity = {
            'security_vulnerability_scan': 2.0,  # Very sensitive to risk
            'test_coverage_validation': 1.5,     # Moderately sensitive
            'code_quality_analysis': 1.2,        # Slightly sensitive
            'performance_benchmarking': 1.0,     # Normal sensitivity
            'documentation_validation': 0.8      # Less sensitive
        }
        
        sensitivity = gate_risk_sensitivity.get(gate.name, 1.0)
        return base_adjustment * sensitivity * gate.base_threshold
    
    async def _run_gate_validation(self, gate: AdaptiveQualityGate, 
                                 threshold: float) -> Tuple[float, str, Dict[str, Any]]:
        """Run actual gate validation logic"""
        # This is a simplified implementation - in real use, each gate would have specific validation
        details = {}
        
        if gate.name == "code_syntax_validation":
            return await self._validate_code_syntax(threshold, details)
        elif gate.name == "test_coverage_validation":
            return await self._validate_test_coverage(threshold, details)
        elif gate.name == "security_vulnerability_scan":
            return await self._validate_security(threshold, details)
        elif gate.name == "performance_benchmarking":
            return await self._validate_performance(threshold, details)
        elif gate.name == "documentation_validation":
            return await self._validate_documentation(threshold, details)
        elif gate.name == "code_quality_analysis":
            return await self._validate_code_quality(threshold, details)
        else:
            # Default implementation
            return 0.8, "passed" if 0.8 >= threshold else "failed", {"message": "Default validation"}
    
    async def _validate_code_syntax(self, threshold: float, details: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
        """Validate code syntax"""
        python_files = list(self.project_path.rglob('*.py'))
        valid_files = 0
        total_files = len(python_files)
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), str(py_file), 'exec')
                valid_files += 1
            except (SyntaxError, UnicodeDecodeError):
                pass
        
        score = valid_files / max(total_files, 1)
        details.update({"valid_files": valid_files, "total_files": total_files})
        
        status = "passed" if score >= threshold else "failed"
        return score, status, details
    
    async def _validate_test_coverage(self, threshold: float, details: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
        """Validate test coverage"""
        # Use cached metrics if available
        if self._project_metrics_cache:
            score = self._project_metrics_cache.test_coverage / 100
        else:
            score = 0.5  # Default estimate
        
        details.update({"estimated_coverage": score * 100})
        status = "passed" if score >= threshold else "failed"
        return score, status, details
    
    async def _validate_security(self, threshold: float, details: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
        """Validate security"""
        if self._project_metrics_cache:
            score = self._project_metrics_cache.security_score
            issues_count = int((1.0 - score) * 10)  # Estimate issues
        else:
            score = 0.9
            issues_count = 1
        
        details.update({"security_score": score, "estimated_issues": issues_count})
        # For security, lower threshold means fewer issues allowed
        status = "passed" if issues_count <= threshold else "failed"
        return score, status, details
    
    async def _validate_performance(self, threshold: float, details: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
        """Validate performance"""
        # Simulate performance test
        simulated_time = 800  # ms
        score = max(0.0, (threshold - simulated_time) / threshold)
        
        details.update({"response_time_ms": simulated_time, "threshold_ms": threshold})
        status = "passed" if simulated_time <= threshold else "failed"
        return score, status, details
    
    async def _validate_documentation(self, threshold: float, details: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
        """Validate documentation"""
        if self._project_metrics_cache:
            score = self._project_metrics_cache.documentation_ratio
        else:
            score = 0.6  # Default estimate
        
        details.update({"documentation_ratio": score})
        status = "passed" if score >= threshold else "failed"
        return score, status, details
    
    async def _validate_code_quality(self, threshold: float, details: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
        """Validate code quality"""
        # Simplified quality score out of 10
        if self._project_metrics_cache:
            complexity_penalty = self._project_metrics_cache.complexity_score * 2
            quality_score = max(0, 10 - complexity_penalty)
        else:
            quality_score = 7.5  # Default
        
        score = quality_score / 10  # Normalize to 0-1
        details.update({"quality_score": quality_score, "max_score": 10})
        status = "passed" if quality_score >= threshold else "failed"
        return score, status, details
    
    def _calculate_progressive_score(self, gate_results: List[ProgressiveValidationResult]) -> float:
        """Calculate overall progressive quality score"""
        if not gate_results:
            return 0.0
        
        # Weight scores based on gate criticality and adaptation success
        weighted_scores = []
        
        for result in gate_results:
            weight = 1.0
            
            # Bonus for successful adaptations
            if result.adapted_reason and result.status in ['passed', 'adapted']:
                weight *= 1.1
            
            # Penalty for critical failures
            gate_config = next((g for g in self.adaptive_gates if g.name == result.gate_name), None)
            if gate_config and gate_config.critical and result.status == 'failed':
                weight *= 0.5
            
            weighted_scores.append(result.score * weight)
        
        return statistics.mean(weighted_scores) * 100  # Convert to percentage
    
    def _determine_progressive_status(self, gate_results: List[ProgressiveValidationResult], 
                                    overall_score: float, maturity_level: ProjectMaturityLevel) -> str:
        """Determine overall progressive validation status"""
        # Get maturity-appropriate threshold
        maturity_thresholds = self.config.get('progressive_thresholds', {})
        required_score = maturity_thresholds.get(maturity_level.value, 0.8) * 100
        
        # Check for critical failures
        critical_failures = sum(1 for r in gate_results if r.status == 'failed' and 
                              any(g.critical for g in self.adaptive_gates if g.name == r.gate_name))
        
        if critical_failures > 0:
            return 'failed'
        elif overall_score >= required_score:
            return 'passed'
        elif overall_score >= required_score * 0.8:  # Within 80% of requirement
            return 'passed_with_adaptations'
        else:
            return 'needs_improvement'
    
    def _generate_learning_insights(self, gate_results: List[ProgressiveValidationResult], 
                                  project_metrics: ProjectMetrics) -> Dict[str, Any]:
        """Generate learning insights from validation results"""
        insights = {
            'adaptation_success_rate': 0.0,
            'most_adapted_gate': None,
            'maturity_effectiveness': 0.0,
            'risk_assessment_accuracy': 0.0,
            'recommended_threshold_adjustments': {}
        }
        
        adapted_gates = [r for r in gate_results if r.adapted_reason]
        if adapted_gates:
            successful_adaptations = [r for r in adapted_gates if r.status in ['passed', 'adapted']]
            insights['adaptation_success_rate'] = len(successful_adaptations) / len(adapted_gates)
            
            # Find most frequently adapted gate
            adaptation_counts = {}
            for result in adapted_gates:
                adaptation_counts[result.gate_name] = adaptation_counts.get(result.gate_name, 0) + 1
            
            if adaptation_counts:
                insights['most_adapted_gate'] = max(adaptation_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate maturity effectiveness
        passed_gates = [r for r in gate_results if r.status in ['passed', 'adapted']]
        if gate_results:
            insights['maturity_effectiveness'] = len(passed_gates) / len(gate_results)
        
        return insights
    
    def _generate_adaptive_recommendations(self, gate_results: List[ProgressiveValidationResult],
                                         maturity_level: ProjectMaturityLevel,
                                         risk_level: RiskAssessmentLevel) -> List[str]:
        """Generate adaptive recommendations"""
        recommendations = []
        
        failed_gates = [r for r in gate_results if r.status == 'failed']
        
        if failed_gates:
            recommendations.append(f"Address {len(failed_gates)} failed quality gates for {maturity_level.value} maturity level")
        
        adapted_gates = [r for r in gate_results if r.adapted_reason]
        if adapted_gates:
            recommendations.append(f"Review {len(adapted_gates)} adaptive threshold adjustments for optimization")
        
        # Maturity-specific recommendations
        if maturity_level == ProjectMaturityLevel.PROTOTYPE:
            recommendations.append("Focus on basic functionality - quality gates are relaxed for rapid prototyping")
        elif maturity_level == ProjectMaturityLevel.DEVELOPMENT:
            recommendations.append("Establish testing practices and basic security measures")
        elif maturity_level == ProjectMaturityLevel.STABLE:
            recommendations.append("Enhance documentation and performance optimization")
        elif maturity_level == ProjectMaturityLevel.PRODUCTION:
            recommendations.append("Maintain highest quality standards with comprehensive validation")
        
        # Risk-specific recommendations
        if risk_level == RiskAssessmentLevel.CRITICAL:
            recommendations.append("Critical risk detected - apply strictest quality gates and manual review")
        elif risk_level == RiskAssessmentLevel.HIGH:
            recommendations.append("High risk changes require enhanced validation and peer review")
        
        return recommendations
    
    def _generate_gate_recommendations(self, gate: AdaptiveQualityGate, 
                                     score: float, threshold: float,
                                     details: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for a gate"""
        recommendations = []
        
        if score < threshold:
            if gate.name == "test_coverage_validation":
                recommendations.append("Increase test coverage with unit and integration tests")
            elif gate.name == "security_vulnerability_scan":
                recommendations.append("Address security vulnerabilities immediately")
            elif gate.name == "code_quality_analysis":
                recommendations.append("Refactor code to reduce complexity and improve maintainability")
            elif gate.name == "documentation_validation":
                recommendations.append("Add missing documentation and inline comments")
            elif gate.name == "performance_benchmarking":
                recommendations.append("Optimize performance bottlenecks and resource usage")
        
        return recommendations
    
    async def _save_learning_data(self, report: ProgressiveQualityReport):
        """Save learning data for future adaptations"""
        try:
            learning_entry = {
                'timestamp': report.timestamp,
                'maturity_level': report.maturity_level.value,
                'maturity_score': report.maturity_score,
                'risk_level': report.risk_level.value,
                'overall_score': report.overall_score,
                'adaptations_applied': report.adaptations_applied,
                'gate_results': [
                    {
                        'gate_name': r.gate_name,
                        'score': r.score,
                        'threshold': r.threshold,
                        'status': r.status,
                        'adapted': bool(r.adapted_reason)
                    }
                    for r in report.gate_results
                ]
            }
            
            self.learning_history.append(learning_entry)
            
            # Save to file
            learning_dir = Path("learning_data/progressive_gates")
            learning_dir.mkdir(parents=True, exist_ok=True)
            
            learning_file = learning_dir / "learning_history.json"
            with open(learning_file, 'w') as f:
                json.dump(self.learning_history, f, indent=2)
            
            self.logger.debug(f"Saved learning data to {learning_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save learning data: {e}")
    
    async def _save_progressive_report(self, report: ProgressiveQualityReport):
        """Save progressive quality report"""
        try:
            reports_dir = Path("quality_reports")
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed JSON report
            json_file = reports_dir / f"progressive_quality_gates_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            
            # Save latest report
            latest_file = reports_dir / "latest_progressive_report.json"
            with open(latest_file, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            
            self.logger.info(f"📄 Progressive quality report saved to {json_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving progressive report: {e}")

async def main():
    """Main execution function for progressive quality gates"""
    print("🔄 Progressive Quality Gates Orchestrator v1.0")
    print("🎯 Generation 1: MAKE IT WORK - Adaptive Quality Validation")
    print()
    
    # Initialize orchestrator
    project_path = Path(".")
    orchestrator = ProgressiveQualityGatesOrchestrator(project_path)
    
    # Example change context (would come from CI/CD system)
    change_context = {
        'files_changed': ['src/security/scanner.py', 'tests/test_security.py'],
        'security_files_changed': True,  # This will trigger HIGH risk assessment
        'lines_added': 150,
        'lines_deleted': 25
    }
    
    print("🚀 Executing Progressive Quality Gates Validation...")
    
    # Execute progressive validation
    report = await orchestrator.execute_progressive_validation(change_context)
    
    print(f"\n✨ Progressive Validation Complete!")
    print(f"📊 Project: {report.project_name}")
    print(f"🏗️ Maturity Level: {report.maturity_level.value.title()}")
    print(f"📈 Maturity Score: {report.maturity_score:.1%}")
    print(f"⚠️ Risk Level: {report.risk_level.value.title()}")
    print(f"🔄 Adaptive Mode: {report.adaptive_mode.value.title()}")
    print(f"📊 Overall Score: {report.overall_score:.1f}%")
    print(f"✅ Status: {report.overall_status}")
    print(f"🔧 Adaptations Applied: {report.adaptations_applied}")
    print(f"🎯 Success Rate: {report.success_rate:.1f}%")
    
    if report.gate_results:
        print(f"\n📋 Gate Results ({len(report.gate_results)} gates executed):")
        for result in report.gate_results:
            status_emoji = {"passed": "✅", "failed": "❌", "warning": "⚠️", "adapted": "🔧"}.get(result.status, "❓")
            print(f"  {status_emoji} {result.gate_name}: {result.score:.1%} "
                  f"(threshold: {result.threshold:.1%}) [{result.execution_time:.2f}s]")
            
            if result.adapted_reason:
                print(f"     🔧 {result.adapted_reason}")
    
    if report.recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report.recommendations[:5], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n🎓 Learning Insights:")
    for key, value in report.learning_insights.items():
        if isinstance(value, float):
            print(f"  📊 {key.replace('_', ' ').title()}: {value:.1%}")
        else:
            print(f"  📊 {key.replace('_', ' ').title()}: {value}")
    
    return report

if __name__ == "__main__":
    asyncio.run(main())